from data_provider.data_factory import data_provider

from utils.tools import EarlyStopping
from utils.metrics import metric

from model9_NS_transformer.ns_models import ns_Transformer
from model9_NS_transformer.exp.exp_basic import Exp_Basic
from model9_NS_transformer.diffusion_models import diffuMTS
from model9_NS_transformer.diffusion_models.diffusion_utils import *
from model9_NS_transformer.diffusion_models.diffusion_utils import (
    q_sample_residual,
    p_sample_loop_residual,
)

import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

from multiprocessing import Pool
import CRPS.CRPS as pscore

import warnings

warnings.filterwarnings("ignore")


def ccc(id, pred, true):
    res_box = np.zeros(len(true))
    for i in range(len(true)):
        res = pscore(pred[i], true[i]).compute()
        res_box[i] = res[0]
    return res_box


def log_normal(x, mu, var):
    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    return 0.5 * torch.mean(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var
    )


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.freeze_base_model = getattr(args, "freeze_base_model", True)

    def _build_model(self):
        model = diffuMTS.Model(self.args, self.device).float()
        cond_pred_model = ns_Transformer.Model(self.args).float()

        # Compatible placeholder for original Exp_Basic return signature.
        cond_pred_model_train = nn.Identity()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            cond_pred_model = nn.DataParallel(cond_pred_model, device_ids=self.args.device_ids)
            cond_pred_model_train = nn.DataParallel(cond_pred_model_train, device_ids=self.args.device_ids)

        return model, cond_pred_model, cond_pred_model_train

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, mode="Model"):
        if mode == "Model":
            params = [{"params": self.model.parameters()}]
            if not self.freeze_base_model:
                params.append({"params": self.cond_pred_model.parameters()})
            model_optim = optim.Adam(params, lr=self.args.learning_rate)
        else:
            model_optim = None
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def _freeze_backbone_if_needed(self):
        if self.freeze_base_model:
            self.cond_pred_model.eval()
            for p in self.cond_pred_model.parameters():
                p.requires_grad = False

    def _forward_base(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """
        ns_Transformer.forward() is expected to return:
            y_base, dec_out, KL_loss, z_sample, enc_feat

        Here we only use y_base and truncate it to pred_len.
        """
        y_base, _, _, _, _ = self.cond_pred_model(
            batch_x, batch_x_mark, dec_inp, batch_y_mark
        )
        y_base = y_base[:, -self.args.pred_len:, :]
        return y_base

    def _sample_timesteps(self, batch_size):
        """
        Same symmetric timestep sampling as the original code.
        """
        t = torch.randint(
            low=0,
            high=self.model.num_timesteps,
            size=(batch_size // 2 + 1,),
        ).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:batch_size]
        return t

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        self.cond_pred_model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1,
                ).float().to(self.device)

                y_base = self._forward_base(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                batch_y_target = batch_y[:, -self.args.pred_len:, :]
                r0 = batch_y_target - y_base.detach()

                # First ablation:
                # r_prior = 0
                # Residual Patch Transformer
                # noise loss
                r_prior = torch.zeros_like(r0).to(self.device)

                n = batch_x.size(0)
                t = self._sample_timesteps(n)

                e = torch.randn_like(r0).to(self.device)
                r_t_batch = q_sample_residual(
                    r0,
                    r_prior,
                    self.model.alphas_bar_sqrt,
                    self.model.one_minus_alphas_bar_sqrt,
                    t,
                    noise=e,
                )

                # IMPORTANT:
                # pass y_base instead of r0.
                output = self.model(batch_x, batch_x_mark, y_base, r_t_batch, r_prior, t)
                output = output[:, -self.args.pred_len:, :]

                loss = (e - output).square().mean()
                total_loss.append(loss.detach().cpu().item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        path_base = os.path.join(path, "best_cond_model_dir")

        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path_base):
            os.makedirs(path_base)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # Optional: load pretrained base model if it exists.
        base_ckpt = os.path.join(path_base, "checkpoint.pth")
        if os.path.exists(base_ckpt):
            print(f"loading pretrained base model from: {base_ckpt}")
            self.cond_pred_model.load_state_dict(
                torch.load(base_ckpt, map_location=self.device),
                strict=False,
            )

        self._freeze_backbone_if_needed()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            iter_count = 0
            train_loss = []

            self.model.train()
            if self.freeze_base_model:
                self.cond_pred_model.eval()
            else:
                self.cond_pred_model.train()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1,
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        y_base = self._forward_base(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                        )

                        batch_y_target = batch_y[:, -self.args.pred_len:, :]
                        r0 = batch_y_target - y_base.detach()

                        r_prior = torch.zeros_like(r0).to(self.device)

                        n = batch_x.size(0)
                        t = self._sample_timesteps(n)

                        e = torch.randn_like(r0).to(self.device)
                        r_t_batch = q_sample_residual(
                            r0,
                            r_prior,
                            self.model.alphas_bar_sqrt,
                            self.model.one_minus_alphas_bar_sqrt,
                            t,
                            noise=e,
                        )

                        # IMPORTANT:
                        # pass y_base instead of r0.
                        output = self.model(batch_x, batch_x_mark, y_base, r_t_batch, r_prior, t)
                        output = output[:, -self.args.pred_len:, :]

                        loss = (e - output).square().mean()
                else:
                    y_base = self._forward_base(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                    )

                    batch_y_target = batch_y[:, -self.args.pred_len:, :]
                    r0 = batch_y_target - y_base.detach()

                    r_prior = torch.zeros_like(r0).to(self.device)

                    n = batch_x.size(0)
                    t = self._sample_timesteps(n)

                    e = torch.randn_like(r0).to(self.device)
                    r_t_batch = q_sample_residual(
                        r0,
                        r_prior,
                        self.model.alphas_bar_sqrt,
                        self.model.one_minus_alphas_bar_sqrt,
                        t,
                        noise=e,
                    )

                    # IMPORTANT:
                    # pass y_base instead of r0.
                    output = self.model(batch_x, batch_x_mark, y_base, r_t_batch, r_prior, t)
                    output = output[:, -self.args.pred_len:, :]

                    loss = (e - output).square().mean()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1,
                            epoch + 1,
                            loss.item(),
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed,
                            left_time,
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    test_loss,
                )
            )

            early_stopping(vali_loss, self.model, path)

            if math.isnan(train_loss):
                break

            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test(self, setting, test=0):
        #####################################################################################################
        ########################## local functions within the class function scope ##########################

        def compute_true_coverage_by_gen_QI(config, dataset_object, all_true_y, all_generated_y):
            n_bins = config.testing.n_bins
            quantile_list = np.arange(n_bins + 1) * (100 / n_bins)

            y_pred_quantiles = np.percentile(
                all_generated_y.squeeze(),
                q=quantile_list,
                axis=1,
            )

            y_true = all_true_y.T
            quantile_membership_array = ((y_true - y_pred_quantiles) > 0).astype(int)
            y_true_quantile_membership = quantile_membership_array.sum(axis=0)

            y_true_quantile_bin_count = np.array(
                [(y_true_quantile_membership == v).sum() for v in np.arange(n_bins + 2)]
            )

            y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
            y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]

            y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]
            y_true_ratio_by_bin = y_true_quantile_bin_count_ / dataset_object

            assert np.abs(np.sum(y_true_ratio_by_bin) - 1) < 1e-10, (
                "Sum of quantile coverage ratios shall be 1!"
            )

            qice_coverage_ratio = np.absolute(
                np.ones(n_bins) / n_bins - y_true_ratio_by_bin
            ).mean()

            return y_true_ratio_by_bin, qice_coverage_ratio, y_true

        def compute_PICP(config, y_true, all_gen_y, return_CI=False):
            low, high = config.testing.PICP_range

            CI_y_pred = np.percentile(
                all_gen_y.squeeze(),
                q=[low, high],
                axis=1,
            )

            y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
            coverage = y_in_range.mean()

            if return_CI:
                return coverage, CI_y_pred, low, high
            else:
                return coverage, low, high

        #####################################################################################################

        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(
                    os.path.join("./checkpoints/" + setting, "checkpoint.pth"),
                    map_location=self.device,
                )
            )

            base_ckpt = os.path.join(
                os.path.join(self.args.checkpoints, setting),
                "best_cond_model_dir",
                "checkpoint.pth",
            )

            if os.path.exists(base_ckpt):
                self.cond_pred_model.load_state_dict(
                    torch.load(base_ckpt, map_location=self.device),
                    strict=False,
                )

        preds = []
        trues = []

        # New: collect y_base point predictions.
        ybase_preds = []
        ybase_trues = []

        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        minibatch_sample_start = time.time()

        self.model.eval()
        self.cond_pred_model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1,
                ).float().to(self.device)

                y_base = self._forward_base(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                r_prior = torch.zeros_like(y_base).to(self.device)

                repeat_n = int(
                    self.model.diffusion_config.testing.n_z_samples
                    / self.model.diffusion_config.testing.n_z_samples_depart
                )

                y_base_tile = y_base.repeat(repeat_n, 1, 1, 1)
                y_base_tile = y_base_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                r_prior_tile = r_prior.repeat(repeat_n, 1, 1, 1)
                r_prior_tile = r_prior_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
                x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
                x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                gen_y_box = []

                for _ in range(self.model.diffusion_config.testing.n_z_samples_depart):
                    # IMPORTANT:
                    # p_sample_loop_residual must accept y_base as condition.
                    r_tile_seq = p_sample_loop_residual(
                        self.model,
                        x_tile,
                        x_mark_tile,
                        y_base_tile,
                        r_prior_tile,
                        self.model.num_timesteps,
                        self.model.alphas,
                        self.model.one_minus_alphas_bar_sqrt,
                    )

                    gen_r = r_tile_seq[-1].reshape(
                        self.args.test_batch_size,
                        int(
                            self.model.diffusion_config.testing.n_z_samples
                            / self.model.diffusion_config.testing.n_z_samples_depart
                        ),
                        self.args.pred_len,
                        self.args.c_out,
                    ).cpu().numpy()

                    y_base_np = y_base_tile.reshape(
                        self.args.test_batch_size,
                        int(
                            self.model.diffusion_config.testing.n_z_samples
                            / self.model.diffusion_config.testing.n_z_samples_depart
                        ),
                        self.args.pred_len,
                        self.args.c_out,
                    ).cpu().numpy()

                    gen_y = gen_r + y_base_np
                    gen_y_box.append(gen_y)

                outputs = np.concatenate(gen_y_box, axis=1)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, :, f_dim:]
                batch_y_eval = batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()

                # New: y_base point forecast, without residual diffusion sampling.
                y_base_eval = y_base[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                ybase_preds.append(y_base_eval)
                ybase_trues.append(batch_y_eval)

                preds.append(outputs)
                trues.append(batch_y_eval)

                if i % 5 == 0 and i != 0:
                    print(
                        "Testing: %d/%d cost time: %f min"
                        % (
                            i,
                            len(test_loader),
                            (time.time() - minibatch_sample_start) / 60,
                        )
                    )
                    minibatch_sample_start = time.time()

        preds = np.array(preds)
        trues = np.array(trues)

        # New: convert y_base results to arrays.
        ybase_preds = np.array(ybase_preds)
        ybase_trues = np.array(ybase_trues)

        preds_save = np.array(preds)
        trues_save = np.array(trues)

        preds_ns = np.array(preds).mean(axis=2)

        print("test shape:", preds_ns.shape, trues.shape)

        preds_ns = preds_ns.reshape(-1, preds_ns.shape[-2], preds_ns.shape[-1])
        trues_ns = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        print("test shape:", preds_ns.shape, trues_ns.shape)

        # New: reshape y_base point predictions to [N, pred_len, C].
        ybase_preds_ns = ybase_preds.reshape(
            -1,
            ybase_preds.shape[-2],
            ybase_preds.shape[-1],
        )
        ybase_trues_ns = ybase_trues.reshape(
            -1,
            ybase_trues.shape[-2],
            ybase_trues.shape[-1],
        )

        print("y_base test shape:", ybase_preds_ns.shape, ybase_trues_ns.shape)

        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds_ns, trues_ns)

        print(
            "NT metrc: mse:{:.4f}, mae:{:.4f} , rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}".format(
                mse,
                mae,
                rmse,
                mape,
                mspe,
            )
        )

        # New: y_base point prediction metrics.
        ybase_mae, ybase_mse, ybase_rmse, ybase_mape, ybase_mspe = metric(
            ybase_preds_ns,
            ybase_trues_ns,
        )

        print(
            "YBase point metrc: mse:{:.4f}, mae:{:.4f} , rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}".format(
                ybase_mse,
                ybase_mae,
                ybase_rmse,
                ybase_mape,
                ybase_mspe,
            )
        )

        preds_flat = preds.reshape(-1, preds.shape[-3], preds.shape[-2] * preds.shape[-1])
        preds_flat = preds_flat.transpose(0, 2, 1)
        preds_flat = preds_flat.reshape(-1, preds_flat.shape[-1])

        trues_flat = trues.reshape(-1, 1, trues.shape[-2] * trues.shape[-1])
        trues_flat = trues_flat.transpose(0, 2, 1)
        trues_flat = trues_flat.reshape(-1, trues_flat.shape[-1])

        y_true_ratio_by_bin, qice_coverage_ratio, y_true = compute_true_coverage_by_gen_QI(
            config=self.model.diffusion_config,
            dataset_object=preds_flat.shape[0],
            all_true_y=trues_flat,
            all_generated_y=preds_flat,
        )

        coverage, _, _ = compute_PICP(
            config=self.model.diffusion_config,
            y_true=y_true,
            all_gen_y=preds_flat,
        )

        print(
            "CARD metrc: QICE:{:.4f}%, PICP:{:.4f}%".format(
                qice_coverage_ratio * 100,
                coverage * 100,
            )
        )

        pred = preds_save.reshape(
            -1,
            preds_save.shape[-3],
            preds_save.shape[-2],
            preds_save.shape[-1],
        )

        true = trues_save.reshape(
            -1,
            trues_save.shape[-2],
            trues_save.shape[-1],
        )

        pool = Pool(processes=32)
        all_res = []

        for i in range(pred.shape[-1]):
            p_in = pred[:, :, :, i]
            p_in = p_in.transpose(0, 2, 1)
            p_in = p_in.reshape(-1, p_in.shape[-1])

            t_in = true[:, :, i]
            t_in = t_in.reshape(-1)

            all_res.append(pool.apply_async(ccc, args=(i, p_in, t_in)))

        p_in = np.sum(pred, axis=-1)
        p_in = p_in.transpose(0, 2, 1)
        p_in = p_in.reshape(-1, p_in.shape[-1])

        t_in = np.sum(true, axis=-1)
        t_in = t_in.reshape(-1)

        CRPS_sum = pool.apply_async(ccc, args=(8, p_in, t_in))

        pool.close()
        pool.join()

        all_res_get = []

        for i in range(len(all_res)):
            all_res_get.append(all_res[i].get())

        all_res_get = np.array(all_res_get)

        CRPS_0 = np.mean(all_res_get, axis=0).mean()
        CRPS_sum = CRPS_sum.get().mean()

        print("CRPS", CRPS_0, "CRPS_sum", CRPS_sum)

        f = open("result.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n")
        f.write("ybase_mse:{}, ybase_mae:{}".format(ybase_mse, ybase_mae))
        f.write("\n\n")
        f.close()

        np.save(
            folder_path + "metrics.npy",
            np.array(
                [
                    mse,
                    mae,
                    rmse,
                    mape,
                    mspe,
                    qice_coverage_ratio * 100,
                    coverage * 100,
                    CRPS_0,
                    CRPS_sum,
                ]
            ),
        )

        np.save(folder_path + "pred.npy", preds_save)
        np.save(folder_path + "true.npy", trues_save)

        # New: save y_base point prediction results.
        np.save(
            folder_path + "ybase_metrics.npy",
            np.array(
                [
                    ybase_mse,
                    ybase_mae,
                    ybase_rmse,
                    ybase_mape,
                    ybase_mspe,
                ]
            ),
        )
        np.save(folder_path + "ybase_pred.npy", ybase_preds)
        np.save(folder_path + "ybase_true.npy", ybase_trues)

        np.save("./results/{}.npy".format(self.args.model_id), np.array(mse))

        np.save(
            "./results/{}_Ntimes.npy".format(self.args.model_id),
            np.array(
                [
                    mse,
                    mae,
                    rmse,
                    mape,
                    mspe,
                    qice_coverage_ratio * 100,
                    coverage * 100,
                    CRPS_0,
                    CRPS_sum,
                ]
            ),
        )

        # New: compact y_base metric files.
        np.save("./results/{}_ybase.npy".format(self.args.model_id), np.array(ybase_mse))
        np.save(
            "./results/{}_ybase_metrics.npy".format(self.args.model_id),
            np.array(
                [
                    ybase_mse,
                    ybase_mae,
                    ybase_rmse,
                    ybase_mape,
                    ybase_mspe,
                ]
            ),
        )

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        preds = []

        self.model.eval()
        self.cond_pred_model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros(
                    [batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]
                ).float().to(self.device)

                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1,
                ).float().to(self.device)

                y_base = self._forward_base(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                r_prior = torch.zeros_like(y_base).to(self.device)

                r_seq = p_sample_loop_residual(
                    self.model,
                    batch_x,
                    batch_x_mark,
                    y_base,
                    r_prior,
                    self.model.num_timesteps,
                    self.model.alphas,
                    self.model.one_minus_alphas_bar_sqrt,
                )

                r0 = r_seq[-1]
                y_pred = y_base + r0

                pred = y_pred[:, -self.args.pred_len:, :].detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "real_prediction.npy", preds)

        return

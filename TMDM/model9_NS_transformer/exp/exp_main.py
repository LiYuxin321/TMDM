from data_provider.data_factory import data_provider

from utils.tools import EarlyStopping
from utils.metrics import metric

from model9_NS_transformer.ns_models import ns_Transformer
from model9_NS_transformer.exp.exp_basic import Exp_Basic
from model9_NS_transformer.diffusion_models import diffuMTS
from model9_NS_transformer.diffusion_models.diffusion_utils import *

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

from multiprocessing import Pool
import CRPS.CRPS as pscore

import warnings


warnings.filterwarnings('ignore')


def ccc(id, pred, true):
    # print(id, datetime.datetime.now())
    res_box = np.zeros(len(true))
    for i in range(len(true)):
        res = pscore(pred[i], true[i]).compute()
        res_box[i] = res[0]
    return res_box


def log_normal(x, mu, var):
    """Logarithm of normal distribution with mean=mu and variance=var
       log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

    Args:
       x: (array) corresponding array containing the input
       mu: (array) corresponding array containing the mean
       var: (array) corresponding array containing the variance

    Returns:
       output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
    """
    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    # return -0.5 * torch.sum(
    #     np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)
    return 0.5 * torch.mean(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = diffuMTS.Model(self.args, self.device).float()
        # model = Transformer.Model(self.args).float()

        cond_pred_model = ns_Transformer.Model(self.args).float()
        cond_pred_model_train = ns_Transformer.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            cond_pred_model = nn.DataParallel(cond_pred_model, device_ids=self.args.device_ids)
            cond_pred_model_train = nn.DataParallel(cond_pred_model_train, device_ids=self.args.device_ids)
        return model, cond_pred_model, cond_pred_model_train

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, mode='Model'):
        if mode == 'Model':
            # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            model_optim = optim.Adam([{'params': self.model.parameters()}, {'params': self.cond_pred_model.parameters()}],
                                     lr=self.args.learning_rate)
        elif mode == 'Cond':
            model_optim = optim.Adam(self.cond_pred_model_train.parameters(), lr=self.args.learning_rate_Cond)
        else:
            model_optim = None
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

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

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        n = batch_x.size(0)
                        t = torch.randint(
                            low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
                        ).to(self.device)
                        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

                        _, y_0_hat_batch, KL_loss, z_sample = self.cond_pred_model(batch_x, batch_x_mark, dec_inp,
                                                                             batch_y_mark)

                        loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))

                        loss_vae_all = loss_vae + self.args.k_z * KL_loss
                        # y_0_hat_batch = z_sample

                        y_T_mean = y_0_hat_batch
                        e = torch.randn_like(batch_y).to(self.device)

                        y_t_batch = q_sample(batch_y, y_T_mean, self.model.alphas_bar_sqrt,
                                             self.model.one_minus_alphas_bar_sqrt, t, noise=e)
                        output = self.model(batch_x, batch_x_mark, batch_y, y_t_batch, y_0_hat_batch, t)

                        loss = (e[:, -self.args.pred_len:, :] - output[:, -self.args.pred_len:, :]).square().mean() + self.args.k_cond * loss_vae_all
                loss = loss.detach().cpu()

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        path2 = os.path.join(path, 'best_cond_model_dir/')
        path2_load = path + '/' + 'checkpoint.pth'

        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(path2):
            os.makedirs(path2)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)


        model_optim = self._select_optimizer()


        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):

            # Training the diffusion part
            epoch_time = time.time()

            iter_count = 0
            train_loss = []
            self.model.train()
            self.cond_pred_model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        n = batch_x.size(0)
                        t = torch.randint(
                            low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
                        ).to(self.device)
                        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

                        _, y_0_hat_batch, KL_loss, z_sample = self.cond_pred_model(batch_x, batch_x_mark, dec_inp,
                                                                             batch_y_mark)

                        loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))

                        loss_vae_all = loss_vae + self.args.k_z * KL_loss
                        # y_0_hat_batch = z_sample

                        y_T_mean = y_0_hat_batch
                        e = torch.randn_like(batch_y).to(self.device)

                        y_t_batch = q_sample(batch_y, y_T_mean, self.model.alphas_bar_sqrt,
                                             self.model.one_minus_alphas_bar_sqrt, t, noise=e)

                        output = self.model(batch_x, batch_x_mark, batch_y, y_t_batch, y_0_hat_batch, t)

                        # loss = (e[:, -self.args.pred_len:, :] - output[:, -self.args.pred_len:, :]).square().mean()
                        loss = (e - output).square().mean() + self.args.k_cond*loss_vae_all

                        train_loss.append(loss.item())

                        if (i + 1) % 100 == 0:
                            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                            speed = (time.time() - time_now) / iter_count
                            left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                            iter_count = 0
                            time_now = time.time()

                        if self.args.use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(model_optim)
                            scaler.update()
                        else:
                            loss.backward()
                            model_optim.step()

                        a = 0
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)


            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            if (math.isnan(train_loss)):
                break

            if early_stopping.early_stop:
                print("Early stopping")
                break



        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model


    def test(self, setting, test=0):
        #####################################################################################################
        ########################## local functions within the class function scope ##########################

        def store_gen_y_at_step_t(config, config_diff, idx, y_tile_seq):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            current_t = self.model.num_timesteps - idx
            gen_y = y_tile_seq[idx].reshape(config.test_batch_size,
                                            int(config_diff.testing.n_z_samples / config_diff.testing.n_z_samples_depart),
                                            (config.label_len + config.pred_len),
                                            config.c_out).cpu().numpy()
            # directly modify the dict value by concat np.array instead of append np.array gen_y to list
            # reduces a huge amount of memory consumption
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y
            else:
                gen_y_by_batch_list[current_t] = np.concatenate([gen_y_by_batch_list[current_t], gen_y], axis=0)
            return gen_y

        def compute_true_coverage_by_gen_QI(config, dataset_object, all_true_y, all_generated_y):
            n_bins = config.testing.n_bins
            quantile_list = np.arange(n_bins + 1) * (100 / n_bins)
            # compute generated y quantiles
            y_pred_quantiles = np.percentile(all_generated_y.squeeze(), q=quantile_list, axis=1)
            y_true = all_true_y.T
            quantile_membership_array = ((y_true - y_pred_quantiles) > 0).astype(int)
            y_true_quantile_membership = quantile_membership_array.sum(axis=0)
            # y_true_quantile_bin_count = np.bincount(y_true_quantile_membership)
            y_true_quantile_bin_count = np.array(
                [(y_true_quantile_membership == v).sum() for v in np.arange(n_bins + 2)])

            # combine true y falls outside of 0-100 gen y quantile to the first and last interval
            y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
            y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
            y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]
            # compute true y coverage ratio for each gen y quantile interval
            # y_true_ratio_by_bin = y_true_quantile_bin_count_ / dataset_object.test_n_samples
            y_true_ratio_by_bin = y_true_quantile_bin_count_ / dataset_object
            assert np.abs(
                np.sum(y_true_ratio_by_bin) - 1) < 1e-10, "Sum of quantile coverage ratios shall be 1!"
            qice_coverage_ratio = np.absolute(np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()
            return y_true_ratio_by_bin, qice_coverage_ratio, y_true

        def compute_PICP(config, y_true, all_gen_y, return_CI=False):
            """
            Another coverage metric.
            """
            low, high = config.testing.PICP_range
            CI_y_pred = np.percentile(all_gen_y.squeeze(), q=[low, high], axis=1)
            # compute percentage of true y in the range of credible interval
            y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
            coverage = y_in_range.mean()
            if return_CI:
                return coverage, CI_y_pred, low, high
            else:
                return coverage, low, high

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))
            self.cond_pred_model.load_state_dict(torch.load(os.path.join(os.path.join(self.args.checkpoints, setting),
                                                                         'best_cond_model_dir/') + '/' + 'checkpoint.pth',
                                                            map_location=self.device))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        minibatch_sample_start = time.time()

        self.model.eval()
        self.cond_pred_model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                gen_y_by_batch_list = [[] for _ in range(self.model.num_timesteps + 1)]
                y_se_by_batch_list = [[] for _ in range(self.model.num_timesteps + 1)]

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)


                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:

                        _, y_0_hat_batch, _, z_sample = self.cond_pred_model(batch_x, batch_x_mark, dec_inp,
                                                                             batch_y_mark)

                        repeat_n = int(
                            self.model.diffusion_config.testing.n_z_samples / self.model.diffusion_config.testing.n_z_samples_depart)
                        y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
                        y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(self.device)
                        y_T_mean_tile = y_0_hat_tile
                        x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
                        x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                        x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
                        x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                        gen_y_box = []
                        for _ in range(self.model.diffusion_config.testing.n_z_samples_depart):
                            for _ in range(self.model.diffusion_config.testing.n_z_samples_depart):
                                y_tile_seq = p_sample_loop(self.model, x_tile, x_mark_tile, y_0_hat_tile, y_T_mean_tile,
                                                           self.model.num_timesteps,
                                                           self.model.alphas, self.model.one_minus_alphas_bar_sqrt)

                            gen_y = store_gen_y_at_step_t(config=self.model.args,
                                                          config_diff=self.model.diffusion_config,
                                                          idx=self.model.num_timesteps, y_tile_seq=y_tile_seq)
                            gen_y_box.append(gen_y)
                        outputs = np.concatenate(gen_y_box, axis=1)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, :, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        batch_y = batch_y.detach().cpu().numpy()

                        pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                        true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                        preds.append(pred)
                        trues.append(true)

                if i % 5 == 0 and i != 0:
                    print('Testing: %d/%d cost time: %f min' % (
                        i, len(test_loader), (time.time() - minibatch_sample_start) / 60))
                    minibatch_sample_start = time.time()


        preds = np.array(preds)
        trues = np.array(trues)

        preds_save = np.array(preds)
        trues_save = np.array(trues)

        preds_ns = np.array(preds).mean(axis=2)
        print('test shape:', preds_ns.shape, trues.shape)
        preds_ns = preds_ns.reshape(-1, preds_ns.shape[-2], preds_ns.shape[-1])
        trues_ns = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds_ns.shape, trues_ns.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds_ns, trues_ns)
        print('NT metrc: mse:{:.4f}, mae:{:.4f} , rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(mse, mae, rmse, mape,
                                                                                                mspe))
        preds = preds.reshape(-1, preds.shape[-3], preds.shape[-2] * preds.shape[-1])
        preds = preds.transpose(0, 2, 1)
        preds = preds.reshape(-1, preds.shape[-1])

        trues = trues.reshape(-1, 1, trues.shape[-2] * trues.shape[-1])
        trues = trues.transpose(0, 2, 1)
        trues = trues.reshape(-1, trues.shape[-1])
        y_true_ratio_by_bin, qice_coverage_ratio, y_true = compute_true_coverage_by_gen_QI(
            config=self.model.diffusion_config, dataset_object=preds.shape[0],
            all_true_y=trues, all_generated_y=preds, )

        coverage, _, _ = compute_PICP(config=self.model.diffusion_config, y_true=y_true, all_gen_y=preds)

        print('CARD metrc: QICE:{:.4f}%, PICP:{:.4f}%'.format(qice_coverage_ratio * 100, coverage * 100))

        pred = preds_save.reshape(-1, preds_save.shape[-3], preds_save.shape[-2], preds_save.shape[-1])
        true = trues_save.reshape(-1, trues_save.shape[-2], trues_save.shape[-1])

        pool = Pool(processes=32)
        all_res = []
        for i in range(pred.shape[-1]):
            p_in = pred[:, :, :, i]
            p_in = p_in.transpose(0, 2, 1)
            p_in = p_in.reshape(-1, p_in.shape[-1])
            t_in = true[:, :, i]
            t_in = t_in.reshape(-1)
            # print(i)
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
        CRPS_sum = CRPS_sum.get()
        CRPS_sum = CRPS_sum.mean()

        print('CRPS', CRPS_0, 'CRPS_sum', CRPS_sum)

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy',
                np.array([mse, mae, rmse, mape, mspe, qice_coverage_ratio * 100, coverage * 100, CRPS_0, CRPS_sum]))
        np.save(folder_path + 'pred.npy', preds_save)
        np.save(folder_path + 'true.npy', trues_save)

        np.save("./results/{}.npy".format(self.args.model_id), np.array(mse))

        np.save("./results/{}_Ntimes.npy".format(self.args.model_id),
                np.array([mse, mae, rmse, mape, mspe, qice_coverage_ratio * 100, coverage * 100, CRPS_0, CRPS_sum]))

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path), map_location=self.device)

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

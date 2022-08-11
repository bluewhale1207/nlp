def _train(self, epoch):
    self.optimizer.zero_grad()
    self.model.train()
    start_time = time.time()
    total_losses = 0
    losses = 0
    y_pred = []
    y_true = []

    for batch_idx, batch_data in enumerate(self.train_data):
        torch.cuda.empty_cache()
        batch_inputs, batch_labels = self.batch2tensor(batch_data)
        batch_preds = self.model(batch_inputs)
        loss = self.criterion(batch_preds, batch_labels)
        loss.backward()
        loss_val = loss.detch().cpu().item()
        total_losses += loss_val
        losses += loss_val
        output_labels = torch.max(batch_preds, dim=1)[1]
        y_pred.extend(output_labels.cpu().numpy().to_list())
        y_true.extend(batch_labels.cpu().numpy().to_list())

        nn.utils.clip_grad_norm(self.optimizer.all_params, max_norm=clip)
        for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
            optimizer.step()
            scheduler.step()
        self.optimizer.zero_grad()
        if batch_idx % log_interval == 0:
            elapsed = time.time() - start_time
            lrs = self.optimizer.get_lr()

            logging.info(
                '| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr {} | loss {:.4f} | s/batch {:.2f}'.format(
                    epoch, self.step, batch_idx, self.batch_num, lrs,
                    losses / log_interval,
                    elapsed / log_interval))

    during_time = time.time() - start_time

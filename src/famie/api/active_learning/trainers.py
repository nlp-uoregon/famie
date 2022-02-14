# define training code reusable for different models here
from .utils import *
import logging
from transformers import AdamW, get_linear_schedule_with_warmup
from datetime import datetime

ensure_dir(LOG_DIR)

for name in logging.root.manager.loggerDict:
    if 'transformers' in name:
        logging.getLogger(name).setLevel(logging.CRITICAL)

logging.disable(logging.WARNING)

logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=os.path.join(LOG_DIR, 'training-log.txt'),
                    filemode='w')
logger = logging.getLogger(__name__)


class ProxyTrainer:
    def __init__(self, config, task, model, dataset):
        self.config = config
        self.task = task
        self.model = model
        self.project_name = None
        if self.config.use_gpu:
            self.model.cuda()

        self.annotated_data = dataset

        ensure_dir(os.path.join(DATABASE_DIR, dataset.project_id))
        self.init_model_param_fpath = os.path.join(DATABASE_DIR, dataset.project_id, 'proxy_init_weights.ckpt')

        self.unlabeled_data = []
        self.signal = PAUSE_MODEL

        # print trainable weights
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('-' * 100)
        logger.info('> proxy model trainable params:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info('>>> {0}: {1}'.format(name, param.shape))
        logger.info(
            'proxy model n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params,
                                                                                     n_nontrainable_params))
        logger.info('-' * 100)

        # optimizer
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if 'task_adapters' in n if
                           p.requires_grad],
                'lr': self.config.adapter_learning_rate, 'weight_decay': self.config.adapter_weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if 'task_adapters' not in n if
                           p.requires_grad],
                'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay
            }
        ]
        self.optimizer = AdamW(params=param_groups)

        self.schedule = get_linear_schedule_with_warmup(self.optimizer,
                                                        num_warmup_steps=self.annotated_data.batch_num * 5,
                                                        num_training_steps=self.annotated_data.batch_num * self.config.proxy_max_epoch)

    def save_weights(self, save_fpath):
        state_dict = {
            'embedding_name': self.config.proxy_embedding_name,
            'hidden_num': self.config.hidden_num,
            'vocabs': self.config.vocabs[self.annotated_data.project_id],
            'weights': {}
        }
        for name, param in self.model.state_dict().items():
            if name.startswith('embedding.xlmr') and 'adapters' not in name:
                continue
            state_dict['weights'][name] = param
        torch.save(state_dict, save_fpath)

    def load_init_weights(self):
        self.model.eval()
        if os.path.exists(self.init_model_param_fpath):
            init_weights = torch.load(self.init_model_param_fpath)['weights']
            for name, param in self.model.state_dict().items():
                if name not in init_weights:
                    init_weights[name] = param

            self.model.load_state_dict(init_weights)

    def receive_signal(self, signal, unlabeled_data, project_name):
        # if signal != PAUSE_MODEL:
        #     print('{} | proxy model received signal={}'.format(datetime.now(), signal))
        self.signal = signal
        self.unlabeled_data = unlabeled_data
        if project_name:
            self.project_name = project_name

    def select_unlabeled_examples(self):
        '''
        Given unlabeled_data, self.model performs selection and save a list of best unlabeled examples to annotate
        '''

        if self.config.selection == 'mnlp':  # uncertainty-based
            # print('{} | mnlp selection is being used'.format(datetime.now()))
            selected_examples = mnlp_sampling(
                self.unlabeled_data,
                self.model,
                self.config.proxy_tokenizer,
                self.config,
                project_id=self.project_name
            )
        elif self.config.selection == 'bertkm':  # diversity-based
            # print('{} | bertkm selection is being used'.format(datetime.now()))
            selected_examples = bertkm_sampling(
                self.unlabeled_data,
                self.model,
                self.config.proxy_tokenizer,
                self.config,
                project_id=self.project_name
            )
        elif self.config.selection == 'badge':  # combination of uncertainty and diversity
            # print('{} | badge selection is being used'.format(datetime.now()))
            selected_examples = badge_sampling(
                self.unlabeled_data,
                self.model,
                self.config.proxy_tokenizer,
                self.config,
                project_id=self.project_name
            )
        else:
            # print('{} | random selection is being used'.format(datetime.now()))
            selected_examples = random_sampling(self.unlabeled_data, self.config)

        with open(os.path.join(DATABASE_DIR, self.project_name, 'selected-unlabeled-data.json'), 'w') as f:
            for d in selected_examples:
                f.write(json.dumps(d) + '\n')

        # print('*' * 20)
        # print('{} | Selected {} examples.'.format(datetime.now(), len(selected_examples)))
        # print('*' * 20)

    def start_listening(self):
        while True:
            time.sleep(LISTEN_TIME)

            if self.signal == STOP_CONTROLLER:
                # print('{} | Stopping proxy model...'.format(datetime.now()))
                break
            elif self.signal != RUN_PROXY:
                continue

            self.load_init_weights()

            # training set
            progress = tqdm.tqdm(total=self.config.proxy_max_epoch, ncols=75,
                                 desc='Retraining proxy model')

            for epoch in range(self.config.proxy_max_epoch):
                torch.cuda.empty_cache()
                if self.signal != RUN_PROXY:
                    # print('{} | proxy model: breaking epoch'.format(datetime.now()))
                    break
                logger.info('proxy model: epoch {}'.format(epoch))

                self.model.train()
                self.optimizer.zero_grad()
                losses = []
                for batch_idx, batch in enumerate(DataLoader(
                        self.annotated_data, batch_size=self.config.batch_size // self.config.accumulate_step,
                        shuffle=True, collate_fn=self.annotated_data.collate_fn)):
                    if self.signal != RUN_PROXY:
                        progress.close()
                        # print('{} | proxy model: breaking step'.format(datetime.now()))
                        break
                    ######################################
                    loss = self.model(batch)
                    loss = loss * (1 / self.config.accumulate_step)

                    loss.backward()

                    if (batch_idx + 1) % self.config.accumulate_step == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clipping)
                        self.optimizer.step()
                        self.schedule.step()
                        self.optimizer.zero_grad()
                        logger.info(
                            'proxy model: {}: step: {}/{}, loss: {}'.format(datetime.now(), batch_idx + 1,
                                                                            self.annotated_data.batch_num,
                                                                            loss.item()))
                        losses.append(loss.item())
                progress.update(1)
                self.model.eval()
            progress.close()
            ############ SELECTION HAPPENS HERE #########
            # print('{} | proxy model: start selecting unlabeled examples...'.format(datetime.now()))
            self.select_unlabeled_examples()
            print('-' * 50)
            # print('{} | proxy model: retraining and selection are done. Paused!'.format(datetime.now()))
            self.signal = RUN_TARGET  # run the target model right after the proxy model finishses.


class TargetTrainer:
    def __init__(self, config, task, model, dataset):
        self.config = config
        self.task = task
        self.model = model
        self.project_name = None
        if self.config.use_gpu:
            self.model.cuda()

        self.annotated_data = dataset

        ensure_dir(os.path.join(DATABASE_DIR, dataset.project_id))
        ensure_dir(os.path.join(OUTPUT_DIR, dataset.project_id))

        self.init_model_param_fpath = os.path.join(DATABASE_DIR, dataset.project_id,
                                                   'target_init_weights.ckpt')  # torch weights

        self.output_model_param_fpath = os.path.join(OUTPUT_DIR, dataset.project_id,
                                                     'target_output_weights.ckpt')  # json weights

        self.signal = PAUSE_MODEL

        # print trainable weights
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('-' * 100)
        logger.info('> target model trainable params:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info('>>> {0}: {1}'.format(name, param.shape))
        logger.info(
            'target model n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params,
                                                                                      n_nontrainable_params))
        logger.info('-' * 100)

        # optimizer
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if 'task_adapters' in n if
                           p.requires_grad],
                'lr': self.config.adapter_learning_rate, 'weight_decay': self.config.adapter_weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if 'task_adapters' not in n if
                           p.requires_grad],
                'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay
            }
        ]
        self.optimizer = AdamW(params=param_groups)

        self.schedule = get_linear_schedule_with_warmup(self.optimizer,
                                                        num_warmup_steps=self.annotated_data.batch_num * 5,
                                                        num_training_steps=self.annotated_data.batch_num * self.config.target_max_epoch)

    def save_weights(self, save_fpath):
        init_state_dict = {
            'project_name': self.annotated_data.project_id,
            'lang': self.annotated_data.lang,
            'embedding_name': self.config.target_embedding_name,
            'hidden_num': self.config.hidden_num,
            'vocabs': self.config.vocabs[self.annotated_data.project_id],
            'weights': {}
        }
        for name, param in self.model.state_dict().items():
            if name.startswith('embedding.xlmr') and 'adapters' not in name:
                continue
            init_state_dict['weights'][name] = param
        torch.save(init_state_dict, save_fpath)

    def save_json_weights(self, save_fpath):
        self.save_weights(save_fpath)
        json_ckpt = convert_ckpt_to_json(save_fpath)
        with open(save_fpath, 'w') as f:
            json.dump(json_ckpt, f)

    def load_init_weights(self):
        self.model.eval()
        if os.path.exists(self.init_model_param_fpath):
            init_weights = torch.load(self.init_model_param_fpath)['weights']
            for name, param in self.model.state_dict().items():
                if name not in init_weights:
                    init_weights[name] = param

            self.model.load_state_dict(init_weights)

    def receive_signal(self, signal, project_name):
        # if signal != PAUSE_MODEL:
        #     print('{} | target model received signal={}'.format(datetime.now(), signal))
        self.signal = signal
        if project_name:
            self.project_name = project_name

    def start_listening(self):
        while True:
            time.sleep(LISTEN_TIME)

            if self.signal == STOP_CONTROLLER:
                # print('{} | Stopping target model...'.format(datetime.now()))
                break
            elif self.signal != RUN_TARGET:
                continue

            self.load_init_weights()

            progress = tqdm.tqdm(total=self.config.target_max_epoch, ncols=75,
                                 desc='Retraining main model')

            for epoch in range(self.config.target_max_epoch):
                torch.cuda.empty_cache()
                if self.signal != RUN_TARGET:
                    # print('{} | target model: breaking epoch'.format(datetime.now()))
                    break
                logger.info('target model: epoch {}'.format(epoch))
                # training set
                self.model.train()
                self.optimizer.zero_grad()
                losses = []
                for batch_idx, batch in enumerate(DataLoader(
                        self.annotated_data, batch_size=self.config.batch_size // self.config.accumulate_step,
                        shuffle=True, collate_fn=self.annotated_data.collate_fn)):
                    if self.signal != RUN_TARGET:
                        progress.close()
                        # print('{} | target model: breaking step'.format(datetime.now()))
                        break
                    ######################################
                    loss = self.model(batch)
                    loss = loss * (1 / self.config.accumulate_step)

                    loss.backward()

                    if (batch_idx + 1) % self.config.accumulate_step == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clipping)
                        self.optimizer.step()
                        self.schedule.step()
                        self.optimizer.zero_grad()
                        logger.info(
                            'target model: {}: step: {}/{}, loss: {}'.format(datetime.now(), batch_idx + 1,
                                                                             self.annotated_data.batch_num,
                                                                             loss.item()))
                        losses.append(loss.item())
                progress.update(1)
                self.model.eval()
            progress.close()
            precompute_distillation(
                self.annotated_data.data,
                self.model,
                self.config,
                self.project_name
            )
            print('Saving trained main model ...')
            self.save_json_weights(self.output_model_param_fpath)
            print('-' * 50)
            # print('{} | target model: precomputing distillation signals is done!'.format(datetime.now()))
            self.signal = PAUSE_MODEL

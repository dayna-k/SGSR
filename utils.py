import os
import os.path
import torch
import sys

class saveData():
    def __init__(self, args):
        self.args = args
        self.save_dir = os.path.join(args.saveDir, args.load)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)
        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')

    def save_model(self, model, epoch):
        torch.save(
            model.state_dict(),
            self.save_dir_model + '/model_lastest.pt')
        torch.save(
            model.state_dict(),
            self.save_dir_model + '/model_' + str(epoch) + '.pt')
        torch.save(
            model,
            self.save_dir_model + '/model_obj.pt')
        torch.save(
            epoch,
            self.save_dir_model + '/last_epoch.pt')

    def save_log(self, log):
        sys.stdout.flush()
        self.logFile.write(log + '\n')
        self.logFile.flush()

    def save_arg(self, args):

        if os.path.exists(self.save_dir + '/config.txt'):
            logFile = open(self.save_dir + '/config.txt', 'a')
            for arg in vars(args):
                logFile.write('{}: {}\n'.format(arg, getattr(args, arg)))
            logFile.write('\n')
            logFile.flush()

        else:
            logFile = open(self.save_dir + '/config.txt', 'w')
            for arg in vars(args):
                logFile.write('{}: {}\n'.format(arg, getattr(args, arg)))
            logFile.write('\n')
            logFile.flush()


    def load_model(self, model):
        model.load_state_dict(torch.load(self.save_dir_model + '/model_lastest.pt'))
        last_epoch = torch.load(self.save_dir_model + '/last_epoch.pt')
        print("load mode_status frmo {}/model_lastest.pt, epoch: {}".format(self.save_dir_model, last_epoch))
        return model, last_epoch

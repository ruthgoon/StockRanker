
import os
import math
import datetime
import random
import numpy as np
#import torchvision.transforms as transforms
import torchaudio.transforms as audiotransforms
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from DBObject import DBObject

""""
NOTE:
    There is a conflict between self.window_size and the padding used to transform the 1D
    tensor into a 2D spectrogram tensor. Because the minimum input size for the torchaudio
    Spectrogram function is [200, 200] there might be some accuracy issues with the final
    model that arise if the window_size is sufficiently smaller than this. This is due to
    the fact that the tensor padding looks like this:
        [201 - window_size / 2] Tensor [201 - window_size / 2]
    So the padding could be larger than the input itself which might pose some accuracy issues
    with the model.
"""


class TimeGAN:
    def __init__(self, log_path=False, log_file=False):
        """
        The TimeGAN Class wraps the pair of generators/discriminators for both
        layers and configures the batch window

        Parameters:
            - window :: The batch window
        """
        self.window_size = False
        self.default_model_save_path = "/home/dan/timegan/models/"
        self.log_path = "/home/dan/timegan/logs/training/" if not log_path else log_path
        self.log_file = log_file
        self.db = DBObject()
        self.logger = ModelLogger(self.log_path, self.log_file)
        # check for cuda
        self.cuda = torch.cuda.is_available()
        self.learning_rate = 0.0002

    def train(self, full=False, period=False, tickers=False, output_path=False, window_size=False, epochs=400, crit_interval=4, learning_rate=False):
        """
        Trains the data and outputs the tensors to the selected `output_path`. Note* the output path must be a directory,
        not a file.

        Parameters:
            - full :: If True, a full retrain will be performed in which the model is initialized from scratch. If False,
                      The model will be loaded and any new data specified in the `period` will be added to the model

            - period :: A list of two date strings (%Y-%M-%d) where the first index is the start date and the second index is
                        the end date (both inclusive) of the period of which to get the training data from. Note the period must
                        be longer than the `window_size` of the class.

            - tickers :: A list of tickers to include in the model

            - output_path :: The output path of the model (defaults to self.default_model_save_path

            - window_size :: The size of the batch window

            - epochs :: The number of training epochs to perform (default 4000)

            - crit_interval :: How many times per batch to train the generator (default 4)

        Returns:
            - save_path :: The full path of the saved checkpoint files for each of the models. The filenames have the following
                           format: `AMD_l1gw15.pt` where `AMD` signifies the ticker, `l1g` signifies that it's the first layer generator
                           and `w15` signifies the window length.
        """
        if not tickers:
            return False

        if period and type(period) != list:
            return False

        if not window_size and not self.window_size:
            return False

        if window_size:
            self.window_size = window_size

        if output_path:
            self.default_model_save_path = output_path

        if type(tickers) == str:
            tickers = [tickers]

        if learning_rate:
            self.learning_rate = learning_rate

        checkpoints = {}
        # the checkpoint file, if there are multiple tickers, is saved with each ticker
        # concatenated with a - (AMD-AAPL-BRKA_l1dw15.pt)
        tick = "-".join(tickers)

        if not full:
            # stores the layers (0, 1) checkpoint files for each ticker (the key)
            checkpoints = {}
            # try finding a file in the model storage directory that matches the given window size and input parameters
            storage_directory = [f for f in os.listdir(
                self.default_model_save_path) if os.path.isfile(os.path.join(self.default_model_save_path, f))]
            checkpoints[tick] = []
            checkpoints[tick].append(
                {"generator": False, "discriminator": False})
            checkpoints[tick].append(
                {"generator": False, "discriminator": False})
            for f in storage_directory:
                params = f.split("_")[1]
                if tick in f and "l1" in params:
                    if "g" in params:
                        checkpoints[tick][0]["generator"] = f
                    elif "d" in params:
                        checkpoints[tick][0]["discriminator"] = f

        else:
            # generate our own checkpoint files bases
            checkpoints[tick] = []
            for i in range(2):
                generator_file = "{}_l{}gw{}.pt".format(
                    tick, i+1, self.window_size)
                discriminator_file = "{}_l{}dw{}.pt".format(
                    tick, i+1, self.window_size)
                checkpoints[tick].append(
                    {"generator": generator_file, "discriminator": discriminator_file})

        # get and format the input data and spectrogram data
        input_tensors = []
        input_specgrams = []
        
        # since the minimum input size for the Spectrogram function is [200, 200] we need
        # to do a two-sided padding of the input
        specgram_padding = int(math.floor(201 - self.window_size / 2))

        for ticker in tickers:
            spec_data = self._get_stock_data(
                ticker, window_size=self.window_size)
            for tensor in spec_data:
                for i in range(tensor.shape[1]):
                    if tensor.shape[0] != self.window_size:
                        continue
                        # do not bother
                    input_tensors.append(tensor[:, i])
                    specgram = audiotransforms.Spectrogram(
                        pad=specgram_padding).cuda()(tensor[:, i])
                    input_specgrams.append(specgram)

        if len(input_specgrams) != len(input_tensors):
            raise Exception("Tensor and specgram input size mismatch")

        # instantiate the models, begin training

        models = {
            "l1g": L1Generator(input_specgrams[0].shape).cuda(),
            "l1d": L1Discriminator(input_specgrams[0].shape).cuda(),
            "l2g": L2Generator(input_specgrams[0].shape, input_tensors[0].shape).cuda(),
            "l2d": L2Discriminator(input_tensors[0].shape).cuda()
        }

        optimizers = {
            "l1g": torch.optim.RMSprop(models["l1g"].parameters(), lr=self.learning_rate),
            "l1d": torch.optim.RMSprop(models["l1d"].parameters(), lr=self.learning_rate),
            "l2g": torch.optim.RMSprop(models["l2g"].parameters(), lr=self.learning_rate),
            "l2d": torch.optim.RMSprop(models["l2d"].parameters(), lr=self.learning_rate)
        }

        # TODO: Write loader if full=False
        if not full:
            raise Exception(
                "Loading in checkpoints for further training is not supported yet")

        # begin the training
        batches_done = 0
        for epoch in range(epochs):

            for x in range(len(input_tensors)):

                tensor = autograd.Variable(input_tensors[x]).float().cuda()
                specgram = autograd.Variable(input_specgrams[x]).cuda()
                optimizers["l1d"].zero_grad()
                optimizers["l2d"].zero_grad()

                z = torch.from_numpy(np.random.normal(
                    0, 1, (specgram.shape[0], specgram.shape[1]))).float().cuda()

                # Train the first layer to get the first loss value
                z = autograd.Variable(z)
                fake_spectrogram = models["l1g"](z).detach()
                fake_spectrogram = fake_spectrogram[0]

                # second 1D->2D layer
                l2_z = autograd.Variable(fake_spectrogram).float().cuda()
                fake_tensor = models["l2g"](l2_z).detach()
                fake_tensor = fake_tensor[:, 0]

                # compute the loss function for l1
                loss_l1d = -torch.mean(models["l1d"](specgram)).cuda()
                loss_l1d = loss_l1d + \
                    torch.mean(models["l1d"](fake_spectrogram)).cuda()

                loss_l2d = -torch.mean(models["l2d"](tensor)).cuda()
                loss_l2d = loss_l2d + \
                    torch.mean(models["l2d"](fake_tensor)).cuda()

                loss_l1d.backward()
                loss_l2d.backward()
                optimizers["l1d"].step()
                optimizers["l2d"].step()

                if x % crit_interval == 0:
                    optimizers["l1g"].zero_grad()
                    optimizers["l2g"].zero_grad()

                    gen_specgrams = models["l1g"](z).detach()
                    gen_tensor = models["l2g"](fake_spectrogram).detach()

                    loss_l1g = - \
                        torch.mean(models["l1d"](gen_specgrams[0])).cuda()
                    loss_l2g = - \
                        torch.mean(models["l2d"](gen_tensor[:, 0])).cuda()

                    loss_l1g.backward()
                    loss_l2g.backward()
                    optimizers["l1g"].step()
                    optimizers["l2g"].step()

                    # log every crit interval
            self.logger.log_epoch(epoch, tick, loss_l1g.item(
            ), loss_l2g.item(), loss_l1d.item(), loss_l2d.item())

        # once done the training, save the generated generators and discriminators
        if full:
            print("Saving...")
            l1gs = checkpoints[tick][0]["generator"]
            l1ds = checkpoints[tick][0]["discriminator"]

            l2gs = checkpoints[tick][1]["generator"]
            l2ds = checkpoints[tick][1]["discriminator"]

            torch.save(models["l1d"].state_dict(),
                       self.default_model_save_path+l1ds)
            torch.save(models["l1g"].state_dict(),
                       self.default_model_save_path+l1gs)

            torch.save(models["l2g"].state_dict(),
                       self.default_model_save_path+l2gs)
            torch.save(models["l2d"].state_dict(),
                       self.default_model_save_path+l2ds)

            print("saved")

    def _get_stock_data(self, tickers, window_size=0, period=False, cuda=True):
        """
        Gets the stock data and automatically cleans and batches it into batches each
        `window_size` big.

        Parameters:
            - tickers :: An array (or string) of ticker(s) of which to get the data from
            - window_size :: The size of each batch window
            - period :: An array of string dates in the form %Y-%m-%d %H:%M:%S where index 0 is the
                        start date and index 1 is the end date
            - cuda :: If True, all training data will be loaded onto VRAM

        Returns:
            - input_dict :: An object containing a list of training tensors for each ticker split
                            into `data_length / window_size` batches
        """

        if type(tickers) == str:
            tickers = [tickers]

        input_dict = {}

        data_limit = "false"
        current_batch = False if window_size == 0 else 0
        if period:
            delta = (datetime.datetime.strptime(period[1], "%Y-%m-%d %H:%M:%S")) - (
                datetime.datetime.strptime(period[0], "%Y-%m-%d %H:%M:%S"))
            data_limit = delta.days + 1

        for ticker in tickers:
            ticker_data = self.db.get_stock_data(
                ticker, limit=data_limit, period=period)
            ticker_data = ticker_data[ticker]
            cleaned_data = []
            keys = ['open', 'high', 'low', 'close']

            for key in keys:
                key_data = [x[key] for x in ticker_data]
                cleaned_data.append(key_data)

        cleaned_data = np.array(cleaned_data)
        cleaned_data = np.transpose(cleaned_data)

        if cuda:
            cleaned_data = torch.from_numpy(cleaned_data).cuda()
        else:
            cleaned_data = torch.from_numpy(cleaned_data)

        if window_size > 0:
            cleaned_data = torch.split(cleaned_data, window_size)
        return cleaned_data


class L1Generator(nn.Module):
    def __init__(self, input_shape, latent_dimensions=False):
        super(L1Generator, self).__init__()
        if not latent_dimensions:
            latent_dimensions = input_shape[1]

        def stacked_layer(input_features, output_features, normalize=True):
            layers = [nn.Linear(input_features, output_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.input_shape = input_shape
        self.model = nn.Sequential(*stacked_layer(latent_dimensions, 128, normalize=False), *stacked_layer(
            128, 256), *stacked_layer(256, 512), *stacked_layer(512, 1024), nn.Linear(1024, int(np.prod(input_shape))), nn.Tanh())

    def forward(self, z):
        val = self.model(z)
        val = val.view(val.shape[0], *self.input_shape)
        return val


class L1Discriminator(nn.Module):
    def __init__(self, image_shape):
        super(L1Discriminator, self).__init__()
        self.image_shape = image_shape
        self.model = nn.Sequential(nn.Linear(image_shape[1], 512), nn.LeakyReLU(
            0.2, inplace=True), nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, 1))

    def forward(self, val):
        flat = val.view(val.shape[0], -1)
        validity = self.model(flat.float())
        return validity


class L2Generator(nn.Module):
    def __init__(self, input_shape, output_shape, latent_dimensions=False):
        """
        Layer 2 takes the 2D Generated Spectrogram output from the Layer 1 Generator
        and uses it as the input to generated a synthetic 1D Timeseries plot which is
        then fed into the L2Discriminator
        """
        super(L2Generator, self).__init__()
        latent_dimensions = input_shape[1] if not latent_dimensions else latent_dimensions
        self.input_shape = input_shape
        self.output_shape = output_shape

        def stacked_layer(input_features, output_features, normalize=True):
            layers = [nn.Linear(input_features, output_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*stacked_layer(latent_dimensions, 128, normalize=False), *stacked_layer(
            128, 256), *stacked_layer(256, 512), *stacked_layer(512, 1024), nn.Linear(1024, self.output_shape[0]), nn.Tanh())

    def forward(self, tensor):
        synthetic_output = self.model(tensor)

        # since this model outputs a 2d tensor we need to reduce
        # it down to a 1d tensor
        synthetic_output = synthetic_output[0]
        return synthetic_output.view(synthetic_output.shape[0], -1)


class L2Discriminator(nn.Module):
    def __init__(self, input_size, latent_dimensions=False):
        super(L2Discriminator, self).__init__()
        if not latent_dimensions:
            latent_dimensions = input_size
        self.input_shape = input_size

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.input_shape[0])), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, output):
        validity = self.model(output)
        return validity


class ModelLogger:
    def __init__(self, save_path, save_file=False):
        """
        The model logger logs each training epoch, failure or exception into the
        specified save_path
        """
        self.path = save_path
        self.files = ["epochs.log", "exceptions.lg"]
        if save_file:
            self.files[0] = save_file

    def log_epoch(self, epoch, ticker, g1l, g2l, d1l, d2l):
        timestamp = datetime.datetime.now()
        has_header = os.path.isfile(self.path+self.files[0])
        with open(self.path+self.files[0], "a+") as f:
            line = "{},{},{},{},{},{},{}\n".format(
                timestamp, ticker, epoch, g1l, g2l, d1l, d2l)
            if not has_header:
                f.write("timestamp,ticker,epoch,g1_loss,g2_loss,d1_loss,d2_loss\n")
            f.write(line)
        return True

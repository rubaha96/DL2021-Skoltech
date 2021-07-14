# Feel free to add functions, classes...

import torch
from torch import nn
import lj_speech
from torchnlp.encoders.text import pad_tensor

def collate_fn(data):
    phonemes = torch.zeros((len(data), 137))
    durations = torch.zeros((len(data), 137))
    spectrograms = torch.zeros((len(data), 80, 920))
    sequence_lengths = torch.zeros(len(data))
    spectrogram_lengths = torch.zeros(len(data))
    
    for idx, sample in enumerate(data):
        phonemes[idx] = torch.FloatTensor(torch.cat((torch.FloatTensor([dict(zip(lj_speech.POSSIBLE_PHONEME_CODES, range(1, 55)))[code] \
                    for code in sample['phonemes_code']]), torch.ones(137 - len(sample['phonemes_code'])))))
        durations[idx] = torch.FloatTensor(pad_tensor(torch.FloatTensor(sample['phonemes_duration'] \
                                            + [920 - sum(sample['phonemes_duration'])]), length=137))
        spectrograms[idx] = torch.FloatTensor(pad_tensor(torch.tensor(sample['spectrogram']), length=920))
        sequence_lengths[idx] = torch.FloatTensor(len(sample['phonemes_code']))
        spectrogram_lengths[idx] = torch.FloatTensor(len(sample['spectrogram'][0]))
    
    return phonemes, durations, spectrograms, sequence_lengths, spectrogram_lengths

class ConvRes(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
              nn.Conv1d(in_channels, out_channels, 5, padding=2),
              nn.ReLU(),
              nn.BatchNorm1d(out_channels),
              nn.Dropout(0.5)       
        )

        self.res = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x) + self.res(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(137, 256)
        self.pos_encoder = PositionalEncoder(137)
        self.attention_block = nn.Sequential(
            nn.MultiheadAttention(256, num_heads=4, dropout=0.15),
            nn.Linear(256, 1024),
            Convresidual(1024, 256),
            nn.Dropout(0.15)
        )
    
    def forward(self, x):
        x = self.embed(x)
        pos_enc = self.pos_encoder(x)
        x = x + pos_enc
        for _ in range(3):
            x = self.attention_block(x)

        return x

class Duration(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(256, 256, num_layers=3,  bidirectional=True)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x, _ = self.lstm(x) 
        x = self.fc(x)
        return x

class Alignment(nn.Module):
    def __init__(self, input):
        super().__init__()
    
    def forward(self, x)
        x = x/2.0 + torch.cumsum(x)
        w = torch.normal(mean=c, std=)/torch.normal(mean=c, std=).sum()
        x = w @ x
        return x

class Decoder(nn.Module):
    def __init__(self):
      super().__init__()

      self.pos_encoder = PositionalEncoding(137)
      self.attention_block = nn.Sequential(
            nn.MultiheadAttention(256, num_heads=4, dropout=0.1),
            nn.Linear(256, 1024),
            Convresidual(1024, 256),
            nn.Dropout(0.1)
        )

      self.conv1 = nn.Conv1d(hid_dim, 80, 1)

      self.postproc = ConvRes(hid_dim, hid_dim, dropout=0.0)
      self.conv2 = nn.Conv1d(hid_dim, 80, 1)

    def forward(self, x):
        x = self.attention_block(x)
        x += self.pos_encoder(x)
        spect1 = self.conv1(x)
        for _ in range(5):
            x  = self.postproc(x)
        spect2 = self.conv2(x)
        return spect1, spect2 

class TTS(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.encoder = Encoder(137, 256).to(device)
        self.alignment = Alignment(256)
        self.duration = Duration(256)
        self.decoder = Decoder(256, 80).to(device)

    def forward(self, phonemes):
        encoder = self.encoder(phonemes)
        duration = self.duration(encoder)
        alignment = self.alignment(encoder, duration)
        decoder = self.decoder(alignment)

        return decoder

def train_tts(dataset_root, num_epochs):
    """
    Train the TTS system from scratch on LJ-Speech-aligned stored at
    `dataset_root` for `num_epochs` epochs and save the best model to
    (!!! 'best' in terms of audio quality!) "./TTS.pth".

    dataset_root:
        `pathlib.Path`
        The argument for `lj_speech.get_dataset()`.
    """

    train_dataset, val_dataset = lj_speech.get_dataset(DATASET_ROOT)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = TTS()
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        for i, (phonemes, durations, spectrograms, sequence_lengths, spectrogram_lengths) in enumerate(train_loader):
            preds = model(phonemes, durations)
            loss = criterion(preds, (spectrograms == 0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

class TextToSpeechSynthesizer:
    """
    Inference-only interface to the TTS model.
    """
    def __init__(self, checkpoint_path):
        """
        Create the TTS model on GPU, loading its weights from `checkpoint_path`.

        checkpoint_path:
            `str`
        """
        self.vocoder = lj_speech.Vocoder()
        ## ...
        ## ...

    def synthesize_from_text(self, text):
        """
        Synthesize text into voice.

        text:
            `str`

        return:
        audio:
            `torch.Tensor` or `numpy.ndarray`, shape == (1, t)
        """
        phonemes = lj_speech.text_to_phonemes(text)
        return self.synthesize_from_phonemes(phonemes)

    def synthesize_from_phonemes(self, phonemes, durations=None):
        """
        Synthesize phonemes into voice.

        phonemes:
            `list` of `str`
            ARPAbet phoneme codes.
        durations:
            `list` of `int`, optional
            Duration in spectrogram frames for each phoneme.
            If given, used for alignment in the model (like during
            training); otherwise, durations are predicted by the duration
            model.

        return:
        audio:
            torch.Tensor or numpy.ndarray, shape == (1, t)
        """
        ## ...
        ## ...

        # spectrogram = ## ...

        return self.vocoder(spectrogram)


def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'TTS.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'TTS.pth'.
        On Linux (in Colab too), use `$ md5sum TTS.pth`.
        On Windows, use `> CertUtil -hashfile TTS.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'TTS.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here
    # Example: `md5_checksum = "747822ca4436819145de8f9e410ca9ca"`
    # Example: `google_drive_link = "https://drive.google.com/file/d/1uEwFPS6Gb-BBKbJIfv3hvdaXZ0sdXtOo/view?usp=sharing"

    return md5_checksum, google_drive_link
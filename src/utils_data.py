import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# collate batch
class collate_batch(object):
    def __init__(self, aggregation=True):
        """
        Initialize the collator with an aggregation option.
        """
        self.aggregation = aggregation

    def __call__(self, batch):
        # collect the data from the batch into Pytorch tensors
        times_list = [time for _, time, _, _, _ in batch]
        times = torch.tensor(np.array(times_list, dtype=np.float32))

        targets_list = [target for _, _, target, _, _ in batch]
        targets = torch.tensor(np.array(targets_list, dtype=np.float32))

        mask1s_list = [mask1 for _, _, _, mask1, _ in batch]
        mask1s = torch.tensor(np.array(mask1s_list, dtype=np.float32))

        mask2s_list = [mask2 for _, _, _, _, mask2 in batch]
        mask2s = torch.tensor(np.array(mask2s_list, dtype=np.float32))

        # aggregation: input size is batch_size x num_sequences x embedding_size (one level of padding)
        # no aggregation: input size is batch_size x num_sequences x num_tokens x embedding_size (two levels of padding)
        if self.aggregation:
            # convert input data to PyTorch tensors and pad sequences
            inputs = [torch.tensor(input, dtype=torch.float32) for input, _, _, _, _ in batch]
        else:
            # convert input data to a PyTorch tensor and pad sequences
            max_length = max(len(sample) for input, _, _, _, _ in batch for sample in input)
            inputs = []
            for input, _, _, _, _ in batch:
                sequences = [torch.tensor(sample, dtype=torch.float32) for sample in input]
                padded_sequence = pad_sequence(sequences, batch_first=True, padding_value=0)
                padded_sequence = torch.nn.functional.pad(padded_sequence, (0, 0, max_length - padded_sequence.size(1), 0))
                inputs.append(padded_sequence)

        # pad the input sequences
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        
        # da eliminare (serve solo per la 'only token attention')
        #inputs = inputs.view(inputs.size(dim=0), inputs.size(dim=1)*inputs.size(dim=2), inputs.size(dim=3))
   
        return inputs, times, targets, mask1s, mask2s 

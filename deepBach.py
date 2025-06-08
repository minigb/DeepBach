"""
@author: Gaetan Hadjeres
"""

import torch
from pathlib import Path
import argparse
from tqdm import tqdm

from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata

from DeepBach.model_manager import DeepBach

N_SAMPLES = 2

def inference_on_trainset(deepbach, dataset, num_iterations=500, output_dir="inference_outputs"):
    """
    For each sample in the training set, generate two harmonizations and save as MIDI.
    """
    # Get the same split as used for training
    train_dl, _, _ = dataset.data_loaders(batch_size=1, split=(0.85, 0.10))

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for idx, (chorale_tensor, metadata_tensor) in tqdm(enumerate(train_dl), total=len(train_dl)):
        # Remove batch dimension
        chorale_tensor = chorale_tensor.squeeze(0)
        metadata_tensor = metadata_tensor.squeeze(0)
        
        dataset.tensor_to_score(chorale_tensor).write('mid', fp = output_dir / f"train_sample_{idx}_input.mid")
        torch.save(chorale_tensor, output_dir / f"train_sample_{idx}_input.pt")
        torch.save(metadata_tensor, output_dir / f"train_sample_{idx}_meta.pt")

        for run in range(N_SAMPLES):
            # Set the seed for reproducibility
            torch.manual_seed(run)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run)

            # Clone to avoid in-place modification
            score, out_tensor_chorale, out_tensor_metadata = deepbach.generation(
            num_iterations=num_iterations,
            sequence_length_ticks=chorale_tensor.shape[1],
            tensor_chorale=chorale_tensor.clone(),
            tensor_metadata=metadata_tensor.clone(),
            voice_index_range=[1,3]
            )

            midi_path = output_dir / f"train_sample_{idx}_run_{run}.mid"
            score.write('midi', fp=str(midi_path))
            print(f"Saved: {midi_path}")

def main():
    parser = argparse.ArgumentParser(description='DeepBach training and generation')
    parser.add_argument('--note_embedding_dim', type=int, default=20,
                        help='size of the note embeddings')
    parser.add_argument('--meta_embedding_dim', type=int, default=20,
                        help='size of the metadata embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of the LSTMs')
    parser.add_argument('--lstm_hidden_size', type=int, default=256,
                        help='hidden size of the LSTMs')
    parser.add_argument('--dropout_lstm', type=float, default=0.5,
                        help='amount of dropout between LSTM layers')
    parser.add_argument('--linear_hidden_size', type=int, default=256,
                        help='hidden size of the Linear layers')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='training batch size')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='number of training epochs')
    parser.add_argument('--train', action='store_true',
                        help='train the specified model for num_epochs')
    parser.add_argument('--num_iterations', type=int, default=500,
                        help='number of parallel pseudo-Gibbs sampling iterations')
    parser.add_argument('--sequence_length_ticks', type=int, default=64,
                        help='length of the generated chorale (in ticks)')

    args = parser.parse_args()
    dataset_manager = DatasetManager()

    metadatas = [
       FermataMetadata(),
       TickMetadata(subdivision=4),
       KeyMetadata()
    ]
    chorale_dataset_kwargs = {
        'voice_ids':      [0, 1, 2, 3],
        'metadatas':      metadatas,
        'sequences_size': 8,
        'subdivision':    4,
        'step':           4
    }
    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
        name='bach_chorales',
        **chorale_dataset_kwargs
        )
    deepbach = DeepBach(
        dataset=bach_chorales_dataset,
        note_embedding_dim=args.note_embedding_dim,
        meta_embedding_dim=args.meta_embedding_dim,
        num_layers=args.num_layers,
        lstm_hidden_size=args.lstm_hidden_size,
        dropout_lstm=args.dropout_lstm,
        linear_hidden_size=args.linear_hidden_size
    )
    
    inference_on_trainset(deepbach, bach_chorales_dataset)

if __name__ == '__main__':
    main()

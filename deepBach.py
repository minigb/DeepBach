"""
@author: Gaetan Hadjeres
"""

import argparse

from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata

from DeepBach.model_manager import DeepBach


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
        'subdivision':    4
    }
    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
        name='bach_chorales',
        **chorale_dataset_kwargs
        )
    dataset = bach_chorales_dataset

    deepbach = DeepBach(
        dataset=dataset,
        note_embedding_dim=args.note_embedding_dim,
        meta_embedding_dim=args.meta_embedding_dim,
        num_layers=args.num_layers,
        lstm_hidden_size=args.lstm_hidden_size,
        dropout_lstm=args.dropout_lstm,
        linear_hidden_size=args.linear_hidden_size
    )

    if args.train:
        deepbach.train(batch_size=args.batch_size,
                       num_epochs=args.num_epochs)
    else:
        deepbach.load()
        deepbach.cuda()

    print('Generation')
    score, tensor_chorale, tensor_metadata = deepbach.generation(
        num_iterations=args.num_iterations,
        sequence_length_ticks=args.sequence_length_ticks,
    )
    score.show('txt')
    score.show()


if __name__ == '__main__':
    main()

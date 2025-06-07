"""
@author: Gaetan Hadjeres
"""

import click
from pathlib import Path

from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata

from DeepBach.model_manager import DeepBach


@click.command()
@click.option('--note_embedding_dim', default=20,
              help='size of the note embeddings')
@click.option('--meta_embedding_dim', default=20,
              help='size of the metadata embeddings')
@click.option('--num_layers', default=2,
              help='number of layers of the LSTMs')
@click.option('--lstm_hidden_size', default=256,
              help='hidden size of the LSTMs')
@click.option('--dropout_lstm', default=0.5,
              help='amount of dropout between LSTM layers')
@click.option('--linear_hidden_size', default=256,
              help='hidden size of the Linear layers')
@click.option('--batch_size', default=256,
              help='training batch size')
@click.option('--num_epochs', default=5,
              help='number of training epochs')
@click.option('--train', is_flag=True,
              help='train the specified model for num_epochs')
@click.option('--num_iterations', default=500,
              help='number of parallel pseudo-Gibbs sampling iterations')
@click.option('--sequence_length_ticks', default=64,
              help='length of the generated chorale (in ticks)')

def inference_on_trainset(deepbach, dataset, num_iterations=500, output_dir="inference_outputs"):
    """
    For each sample in the training set, generate two harmonizations and save as MIDI.
    """
    # Get the same split as used for training
    train_dl, _, _ = dataset.data_loaders(batch_size=1, split=(0.85, 0.10))

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for idx, (chorale_tensor, metadata_tensor) in enumerate(train_dl):
        # Remove batch dimension
        chorale_tensor = chorale_tensor.squeeze(0)
        metadata_tensor = metadata_tensor.squeeze(0)

        for run in range(2):
            # Clone to avoid in-place modification
            score, out_tensor_chorale, out_tensor_metadata = deepbach.generation(
                num_iterations=num_iterations,
                sequence_length_ticks=chorale_tensor.shape[1],
                tensor_chorale=chorale_tensor.clone(),
                tensor_metadata=metadata_tensor.clone(),
            )

            midi_path = output_dir / f"train_sample_{idx}_run_{run}.mid"
            score.write('midi', fp=str(midi_path))
            print(f"Saved: {midi_path}")

def main(note_embedding_dim,
         meta_embedding_dim,
         num_layers,
         lstm_hidden_size,
         dropout_lstm,
         linear_hidden_size,
         batch_size,
         num_epochs,
         train,
         num_iterations,
         sequence_length_ticks,
         ):
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
        note_embedding_dim=note_embedding_dim,
        meta_embedding_dim=meta_embedding_dim,
        num_layers=num_layers,
        lstm_hidden_size=lstm_hidden_size,
        dropout_lstm=dropout_lstm,
        linear_hidden_size=linear_hidden_size
    )
    
    # inference_on_trainset(deepbach, bach_chorales_dataset)

    # if train:
    #     deepbach.train(batch_size=batch_size,
    #                    num_epochs=num_epochs)
    # else:
    #     deepbach.load()
    #     deepbach.cuda()

    # print('Generation')
    # score, tensor_chorale, tensor_metadata = deepbach.generation(
    #     num_iterations=num_iterations,
    #     sequence_length_ticks=sequence_length_ticks,
    # )
    # score.show('txt')
    # score.show()


if __name__ == '__main__':
    main()

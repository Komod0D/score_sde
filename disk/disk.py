"""disk dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

# TODO(disk): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(disk): BibTeX citation
_CITATION = """
"""


class Disk(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for disk dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(disk): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Tensor(shape=(2, ), dtype=tf.float16),
            'label': tfds.features.ClassLabel(names=['inside', 'outside']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage=None,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):

    spec = {'image': tf.TensorSpec(shape=(2,), dtype=tf.float32, name=None),
          'label': tf.TensorSpec(shape=(), dtype=tf.string, name=None)}
    ds = tf.data.experimental.load('/home/komodo/Documents/uni/thesis/disk/data', spec)
    return {'train': ds}
  
    """Returns SplitGenerators."""
    # TODO(disk): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(disk): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'train_imgs'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(disk): Yields (key, example) tuples from the dataset
    for f in path.glob('*.jpeg'):
      yield 'key', {
          'image': f,
          'label': 'yes',
      }

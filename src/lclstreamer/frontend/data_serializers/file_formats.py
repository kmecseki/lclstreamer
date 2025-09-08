import sys
from collections.abc import Iterator
from io import BytesIO
from typing import Any

import h5py  # type: ignore
import hdf5plugin  # type: ignore

from ...models.parameters import DataSerializerParameters
from ...protocols.backend import StrFloatIntNDArray
from ...protocols.frontend import DataSerializerProtocol
from ...utils.logging_utils import log


class Hdf5BinarySerializer(DataSerializerProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(self, parameters: DataSerializerParameters):
        """
        Initializes an HDF5 data serializer

        This serializers turns a dictionary of numpy arrays into a binary blob with the
        internal structure of an HDF5 file, according to the preferences specified by
        the configuration parameters.

        Arguments:

            parameters: The configuration parameters
        """
        if parameters.Hdf5BinarySerializer is None:
            log.error("No configuration parameters found for Hdf5BinarySerializer")
            sys.exit(1)

        if parameters.Hdf5BinarySerializer.compression == "gzip":
            self._compression_options: dict[str, Any] = {
                "compression": "gzip",
                "compression_opts": parameters.Hdf5BinarySerializer.compression_level,
                "shuffle": False,
            }
        elif parameters.Hdf5BinarySerializer.compression == "gzip_with_shuffle":
            self._compression_options = {
                "compression": "gzip",
                "compression_opts": parameters.Hdf5BinarySerializer.compression_level,
                "shuffle": True,
            }
        elif parameters.Hdf5BinarySerializer.compression == "bitshuffle_with_lz4":
            self._compression_options = {
                "compression": hdf5plugin.Bitshuffle(
                    cname="lz4",
                    clevel=parameters.Hdf5BinarySerializer.compression_level,
                )
            }
        elif parameters.Hdf5BinarySerializer.compression == "bitshuffle_with_zstd":
            self._compression_options = {
                "compression": hdf5plugin.Bitshuffle(
                    cname="zstd",
                    clevel=parameters.Hdf5BinarySerializer.compression_level,
                )
            }
        elif parameters.Hdf5BinarySerializer.compression == "zfp":
            self._compression_options = {"compression": hdf5plugin.Zfp()}
        else:
            self._compression_options = {}

        self._hdf5_fields: dict[str, str] = parameters.Hdf5BinarySerializer.fields

    def __call__(
        self, stream: Iterator[dict[str, StrFloatIntNDArray | None]]
    ) -> Iterator[bytes]:
        """
        Serializes data to a binary blob with an internal HDF5 structure

        Arguments:

            data: A dictionary storing numpy arrays

        Returns

            byte_block: A binary blob (a bytes object)
        """
        data: dict[str, StrFloatIntNDArray | None]
        for data in stream:

            depth_of_data_blocks: list[int] = [
                value.shape[0]
                for data_block in data
                if (value := data[data_block]) is not None
            ]

            if len(set(depth_of_data_blocks)) != 1:
                log.error(
                    "The data blocks that should be written to the HDF5 file have"
                    "different depths"
                )
                sys.exit(1)

            mismatching_entries: set[str] = data.keys() - self._hdf5_fields.keys()

            if len(mismatching_entries) != 0:
                log.error(
                    "The Hdf5BinarySerializer is asked to serialize the following data "
                    "entries but data for these entries is not available: "
                    f"{' '.join(list(mismatching_entries))}"
                )
                sys.exit(1)

            with BytesIO() as byte_block:
                with h5py.File(byte_block, "w") as fh:
                    data_block_name: str
                    for data_block_name in data:
                        if (
                            data_block_name in self._hdf5_fields
                            and (data_block := data[data_block_name]) is not None
                        ):
                            fh.create_dataset(
                                name=self._hdf5_fields[data_block_name],
                                shape=data_block.shape,
                                dtype=data_block.dtype,
                                chunks=(1,) + data_block[0].shape,
                                data=data_block,
                                **self._compression_options,
                            )

                yield byte_block.getvalue()

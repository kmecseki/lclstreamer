import sys
from typing import Any, Callable

import numpy
from numpy.typing import NDArray
from psana import Detector, EventId  # type: ignore

from ...models.parameters import DataSourceParameters
from ...protocols.backend import DataSourceProtocol
from ...utils.logging_utils import log


class Psana1Timestamp(DataSourceProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(
        self,
        name: str,
        parameters: DataSourceParameters,
        additional_info: dict[str, Any],
    ):
        """
        Initializes a psana1 Timestamp data source.

        Arguments:

            name: An identifier for the data source

            parameters: The configuration parameters
        """
        del name
        del parameters
        del additional_info

    def get_data(self, event: Any) -> NDArray[numpy.float_]:
        """
        Retrieves timestamp information from a psana1 event

        Arguments:

            event: A psana1 event

        Returns:

            timestamp: a 1D numpy array (of type float64) containing the timestamp
            information
        """
        psana_event_id: Any = event.get(
            EventId  # pyright: ignore[reportAttributeAccessIssue]
        )
        timestamp_epoch_format: Any = psana_event_id.time()
        return numpy.array(
            numpy.float64(
                str(timestamp_epoch_format[0]) + "." + str(timestamp_epoch_format[1])
            )
        )

class Psana1DetectorInterface(DataSourceProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(
        self,
        name: str,
        parameters: DataSourceParameters,
        additional_info: dict[str, Any],
    ):
        """
        Initializes a psana1 Detector values data source.

        Arguments:

            name: An identifier for the data source

            parameters: The configuration parameters
        """
        extra_parameters: dict[str, Any] | None = parameters.__pydantic_extra__

        self._name: str = name
        if extra_parameters is None:
            log.error(f"Entries needed by the {name} data source are not defined")
            sys.exit(1)
        if "psana_name" not in extra_parameters:
            log.error(f"Entry 'psana_name' is not defined for data source {name}")
            sys.exit(1)
        if "psana_fields" not in extra_parameters:
            if ":" in extra_parameters["psana_name"]:
                self._is_pv: bool = True
            else:
                log.error(f"Entry 'psana_fields' is not defined for data source {name}")
                sys.exit(1)
        else:
            fields: list[str] | str = extra_parameters["psana_fields"]
            self._det_params: list[str] = [fields] if isinstance(fields, str) else fields

        self._detector_interface: Any = Detector(extra_parameters["psana_name"])

    def get_data(self, event: Any) -> NDArray[object]:
        """
        Retrieves Detector values from a psana1 event

        Arguments:

            event: A psana1 event

         Returns:

            value: The retrieved data is a list of object, such as:
            [Object1, Object2, Object3, ...]
            in the format of a numpy array.
        """

        data: list[Any] = []

        base: Any
        if getattr(self, "_is_pv", False):
            base = self._detector_interface
            data.append(base(event))
        else:
            for param in self._det_params:
                base = self._detector_interface
                subfields: list[str] = param.split(".")
                for field in subfields:
                    if hasattr(base, field):
                        base = getattr(base, field)
                        if callable(base):
                            try:
                                base = base(event)
                            except TypeError:
                                base = base()
                    else:
                        log.error(f"Detector {base} has no parameter {field}")
                        sys.exit(1)
                    data.append(base)

        if len(data) == 1:
            data = data[0]
            if isinstance(data, dict):
                log.error(f"Data is in dict format: {self._name}!")
                exit(1)
            else:
                return numpy.array(data, dtype=numpy.float_)
        return numpy.array(data, dtype=object)


class Psana1EvrCodes(DataSourceProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(
        self,
        *,
        name: str,
        parameters: DataSourceParameters,
        additional_info: dict[str, Any],
    ):
        """
        Intializes a psana1 EVR data source

        Arguments:
            name: An identifier for the data source

            parameters: The configuration parameters
        """
        del additional_info
        extra_parameters: dict[str, Any] | None = parameters.__pydantic_extra__
        if extra_parameters is None:
            log.error(f"Entries needed by the {name} data source are not defined")
            sys.exit(1)
        if "psana_name" not in extra_parameters:
            log.error(f"Entry 'psana_name' is not defined for data source {name}")
            sys.exit(1)

        self._detector_interface: Any = Detector(extra_parameters["psana_name"])

    def get_data(self, event: Any) -> NDArray[numpy.int_]:
        """
        Retrieves IpmDetector data from an event

        Arguments:

            event: A psana1 event

        Returns:

            value: A numpy array storing all the EVR codes associated with and
            event (max 256 event codes)
        """
        evr_codes: Any = self._detector_interface.eventCodes(event)
        if evr_codes is None:
            return numpy.ndarray([0] * 256, dtype=numpy.int_)

        current_evr_codes: NDArray[numpy.int_] = numpy.array(
            evr_codes, dtype=numpy.int_
        )

        return numpy.pad(
            current_evr_codes,
            pad_width=(0, 256 - len(current_evr_codes)),
            mode="constant",
            constant_values=(0, 0),
        )

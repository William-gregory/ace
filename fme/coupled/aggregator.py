import dataclasses
import os
import warnings
from collections.abc import Callable, Mapping, Sequence

import torch
import xarray as xr

from fme.ace.aggregator.inference.main import (
    InferenceAggregator as InferenceAggregator_,
)
from fme.ace.aggregator.inference.main import (
    InferenceEvaluatorAggregator as InferenceEvaluatorAggregator_,
)
from fme.ace.aggregator.inference.main import StepMeanEntry
from fme.ace.aggregator.one_step.main import OneStepAggregator as OneStepAggregator_
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.generics.aggregator import (
    AggregatorABC,
    InferenceAggregatorABC,
    InferenceLog,
    InferenceLogs,
)
from fme.core.typing_ import TensorDict, TensorMapping
from fme.coupled.data_loading.batch_data import (
    CoupledPairedData,
    CoupledPrognosticState,
)
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.stepper import CoupledTrainOutput
from fme.coupled.typing_ import CoupledTensorMapping


class TrainAggregator(AggregatorABC[CoupledTrainOutput]):
    def __init__(self):
        self._n_batches = 0
        self._loss = torch.tensor(0.0, device=get_device())

    @torch.no_grad()
    def record_batch(self, batch: CoupledTrainOutput):
        self._loss += batch.total_metrics["loss"]
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, torch.Tensor]:
        logs = {f"{label}/mean/loss": self._loss / self._n_batches}
        dist = Distributed.get_instance()
        for key in sorted(logs.keys()):
            logs[key] = float(dist.reduce_mean(logs[key].detach()).cpu().numpy())
        return logs

    @torch.no_grad()
    def flush_diagnostics(self, subdir: str | None):
        pass


class OneStepAggregator(AggregatorABC[CoupledTrainOutput]):
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        dataset_info: CoupledDatasetInfo,
        loss_scaling: CoupledTensorMapping,
        save_diagnostics: bool = True,
        output_dir: str | None = None,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
        ocean_channel_mean_names: Sequence[str] | None = None,
        ice_channel_mean_names: Sequence[str] | None = None,
        atmosphere_channel_mean_names: Sequence[str] | None = None,
    ):
        """
        Args:
            dataset_info: Coordinate information of dataset.
            save_diagnostics: Whether to save diagnostics to disk.
            output_dir: Directory to write diagnostics to.
            variable_metadata: Metadata for each variable.
            loss_scaling: Optional coupled mapping of variables and their
                scaling factors used in loss computation for the stepper.
            ocean_channel_mean_names: Names to include in ocean channel-mean metrics.
            ice_channel_mean_names: Names to include in ice channel-mean metrics.
            atmosphere_channel_mean_names: Names to include in atmosphere channel-mean
                metrics.

        """
        self._dist = Distributed.get_instance()
        self._loss = torch.tensor(0.0, device=get_device())
        self._loss_ocean = torch.tensor(0.0, device=get_device())
        self._loss_ice = torch.tensor(0.0, device=get_device())
        self._loss_atmos = torch.tensor(0.0, device=get_device())
        self._n_batches = 0
        self._aggregators = {}
        if dataset_info.ocean is not None:
            self._aggregators["ocean"] = OneStepAggregator_(
                dataset_info.ocean,
                save_diagnostics=save_diagnostics,
                output_dir=(
                    os.path.join(output_dir, "ocean")
                    if output_dir is not None
                    else None
                ),
                loss_scaling=loss_scaling.ocean,
                channel_mean_names=ocean_channel_mean_names,
            )
        if dataset_info.ice is not None:
            self._aggregators["ice"] = OneStepAggregator_(
                dataset_info.ice,
                save_diagnostics=save_diagnostics,
                output_dir=(
                    os.path.join(output_dir, "ice")
                    if output_dir is not None
                    else None
                ),
                loss_scaling=loss_scaling.ice,
                channel_mean_names=ice_channel_mean_names,
            )
        if dataset_info.atmosphere is not None:
            self._aggregators["atmosphere"] = OneStepAggregator_(
                dataset_info.atmosphere,
                save_diagnostics=save_diagnostics,
                output_dir=(
                    os.path.join(output_dir, "atmosphere")
                    if output_dir is not None
                    else None
                ),
                loss_scaling=loss_scaling.atmosphere,
                channel_mean_names=atmosphere_channel_mean_names,
            )
        self._num_channels_ocean: int | None = None
        if ocean_channel_mean_names is not None:
            self._num_channels_ocean = len(ocean_channel_mean_names)
        self._num_channels_ice: int | None = None
        if ice_channel_mean_names is not None:
            self._num_channels_ice = len(ice_channel_mean_names)
        self._num_channels_atmos: int | None = None
        if atmosphere_channel_mean_names is not None:
            self._num_channels_atmos = len(atmosphere_channel_mean_names)

    @torch.no_grad()
    def record_batch(
        self,
        batch: CoupledTrainOutput,
    ):
        if self._num_channels_ocean is None:
            self._num_channels_ocean = len(batch.ocean.gen_data)
        if self._num_channels_ice is None:
            self._num_channels_ice = len(batch.ice.gen_data)
        if self._num_channels_atmos is None:
            self._num_channels_atmos = len(batch.atmosphere.gen_data)
        self._loss += batch.total_metrics["loss"]
        if "ocean" in self._aggregators:
            self._aggregators["ocean"].record_batch(batch.ocean)
            self._loss_ocean += batch.ocean.metrics["loss/ocean"]
        if "ice" in self._aggregators:
            self._aggregators["ice"].record_batch(batch.ice)
            self._loss_ice += batch.ice.metrics["loss/ice"]
        if "atmosphere" in self._aggregators:
            self._aggregators["atmosphere"].record_batch(batch.atmosphere)
            self._loss_atmos += batch.atmosphere.metrics["loss/atmosphere"]
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        prefix = f"{label}/mean_norm/weighted_rmse"
        if "atmosphere" not in self._aggregators:
            if self._num_channels_ocean is None or self._num_channels_ice is None:
                raise ValueError("No data recorded.")
            ocean_logs = self._aggregators["ocean"].get_logs(label)
            ocean_logs.pop(f"{label}/mean/loss")
            ocean_logs[f"{prefix}/ocean_channel_mean"] = ocean_logs.pop(
                f"{prefix}/channel_mean"
            )
            ice_logs = self._aggregators["ice"].get_logs(label)
            ice_logs.pop(f"{label}/mean/loss")
            ice_logs[f"{prefix}/ice_channel_mean"] = ice_logs.pop(
                f"{prefix}/channel_mean"
            )
            duplicates = set(ocean_logs.keys()) & set(ice_logs.keys())
            if len(duplicates) > 0:
                raise ValueError(
                    "Duplicate keys found in ocean and ice"
                    f"{label} logs: {duplicates}."
                )
            logs = {**ocean_logs, **ice_logs}
            loss = self._loss / self._n_batches
            logs[f"{label}/mean/loss"] = float(
                self._dist.reduce_mean(loss.detach()).cpu().numpy()
            )
            loss_ocean = self._loss_ocean / self._n_batches
            logs[f"{label}/mean/loss/ocean"] = float(
                self._dist.reduce_mean(loss_ocean.detach()).cpu().numpy()
            )
            loss_ice = self._loss_ice / self._n_batches
            logs[f"{label}/mean/loss/ice"] = float(
                self._dist.reduce_mean(loss_ice.detach()).cpu().numpy()
            )
        elif "ice" not in self._aggregators:
            if self._num_channels_ocean is None or self._num_channels_atmos is None:
                raise ValueError("No data recorded.")
            ocean_logs = self._aggregators["ocean"].get_logs(label)
            ocean_logs.pop(f"{label}/mean/loss")
            ocean_logs[f"{prefix}/ocean_channel_mean"] = ocean_logs.pop(
                f"{prefix}/channel_mean"
            )
            atmos_logs = self._aggregators["atmosphere"].get_logs(label)
            atmos_logs.pop(f"{label}/mean/loss")
            atmos_logs[f"{prefix}/atmosphere_channel_mean"] = atmos_logs.pop(
                f"{prefix}/channel_mean"
            )
            duplicates = set(ocean_logs.keys()) & set(atmos_logs.keys())
            if len(duplicates) > 0:
                raise ValueError(
                    "Duplicate keys found in ocean and atmosphere "
                    f"{label} logs: {duplicates}."
                )
            logs = {**ocean_logs, **atmos_logs}
            loss = self._loss / self._n_batches
            logs[f"{label}/mean/loss"] = float(
                self._dist.reduce_mean(loss.detach()).cpu().numpy()
            )
            loss_ocean = self._loss_ocean / self._n_batches
            logs[f"{label}/mean/loss/ocean"] = float(
                self._dist.reduce_mean(loss_ocean.detach()).cpu().numpy()
            )
            loss_atmos = self._loss_atmos / self._n_batches
            logs[f"{label}/mean/loss/atmosphere"] = float(
                self._dist.reduce_mean(loss_atmos.detach()).cpu().numpy()
            )
        elif "ocean" not in self._aggregators:
            if self._num_channels_ice is None or self._num_channels_atmos is None:
                raise ValueError("No data recorded.")
            ice_logs = self._aggregators["ice"].get_logs(label)
            ice_logs.pop(f"{label}/mean/loss")
            ice_logs[f"{prefix}/ice_channel_mean"] = ice_logs.pop(
                f"{prefix}/channel_mean"
            )
            atmos_logs = self._aggregators["atmosphere"].get_logs(label)
            atmos_logs.pop(f"{label}/mean/loss")
            atmos_logs[f"{prefix}/atmosphere_channel_mean"] = atmos_logs.pop(
                f"{prefix}/channel_mean"
            )
            duplicates = set(ice_logs.keys()) & set(atmos_logs.keys())
            if len(duplicates) > 0:
                raise ValueError(
                    "Duplicate keys found in ice and atmosphere "
                    f"{label} logs: {duplicates}."
                )
            logs = {**ice_logs, **atmos_logs}
            loss = self._loss / self._n_batches
            logs[f"{label}/mean/loss"] = float(
                self._dist.reduce_mean(loss.detach()).cpu().numpy()
            )
            loss_ice = self._loss_ice / self._n_batches
            logs[f"{label}/mean/loss/ice"] = float(
                self._dist.reduce_mean(loss_ice.detach()).cpu().numpy()
            )
            loss_atmos = self._loss_atmos / self._n_batches
            logs[f"{label}/mean/loss/atmosphere"] = float(
                self._dist.reduce_mean(loss_atmos.detach()).cpu().numpy()
            )
        else:
            if (self._num_channels_ice is None
                or self._num_channels_atmos is None
                or self._num_channels_ocean is None):
                raise ValueError("No data recorded.")
            ocean_logs = self._aggregators["ocean"].get_logs(label)
            ocean_logs.pop(f"{label}/mean/loss")
            ocean_logs[f"{prefix}/ocean_channel_mean"] = ocean_logs.pop(
                f"{prefix}/channel_mean"
            )
            ice_logs = self._aggregators["ice"].get_logs(label)
            ice_logs.pop(f"{label}/mean/loss")
            ice_logs[f"{prefix}/ice_channel_mean"] = ice_logs.pop(
                f"{prefix}/channel_mean"
            )
            atmos_logs = self._aggregators["atmosphere"].get_logs(label)
            atmos_logs.pop(f"{label}/mean/loss")
            atmos_logs[f"{prefix}/atmosphere_channel_mean"] = atmos_logs.pop(
                f"{prefix}/channel_mean"
            )
            duplicates = set(ice_logs.keys()) & set(atmos_logs.keys()) & set(ocean_logs.keys())
            if len(duplicates) > 0:
                raise ValueError(
                    "Duplicate keys found in ice, ocean, and atmosphere "
                    f"{label} logs: {duplicates}."
                )
            logs = {**ice_logs, **atmos_logs, **ocean_logs}
            loss = self._loss / self._n_batches
            logs[f"{label}/mean/loss"] = float(
                self._dist.reduce_mean(loss.detach()).cpu().numpy()
            )
            loss_ice = self._loss_ice / self._n_batches
            logs[f"{label}/mean/loss/ice"] = float(
                self._dist.reduce_mean(loss_ice.detach()).cpu().numpy()
            )
            loss_atmos = self._loss_atmos / self._n_batches
            logs[f"{label}/mean/loss/atmosphere"] = float(
                self._dist.reduce_mean(loss_atmos.detach()).cpu().numpy()
            )
            loss_ocean = self._loss_ocean / self._n_batches
            logs[f"{label}/mean/loss/ocean"] = float(
                self._dist.reduce_mean(loss_ocean.detach()).cpu().numpy()
            )

        return logs

    @torch.no_grad()
    def flush_diagnostics(self, subdir: str | None = None):
        """
        Flushes diagnostics to netCDF files, in separate directories for ocean, 
        ice, and atmosphere.
        """
        for aggregator in self._aggregators.values():
            aggregator.flush_diagnostics(subdir)


@dataclasses.dataclass
class InferenceEvaluatorAggregatorConfig:
    """
    Configuration for coupled inference evaluator aggregator.

    Parameters:
        log_histograms: Whether to log histograms of the targets and predictions.
        log_video: Whether to log videos of the state evolution.
        log_extended_video: Whether to log wandb videos of the predictions with
            statistical metrics, only done if log_video is True.
        log_zonal_mean_images: Whether to log zonal-mean images (hovmollers) with a
            time dimension.
        log_seasonal_means: Whether to log seasonal mean metrics and images.
        log_global_mean_time_series: Whether to log global mean time series metrics.
        log_global_mean_norm_time_series: Whether to log the normalized global mean
            time series metrics.
        monthly_reference_data: Path to monthly reference data to compare against.
            This should include both ocean and atmosphere variables.
        time_mean_reference_data: Path to reference time means to compare against.
            This should include both ocean and atmosphere variables.
    """

    log_histograms: bool = False
    log_video: bool = False
    log_extended_video: bool = False
    log_zonal_mean_images: bool = False
    log_seasonal_means: bool = False
    log_global_mean_time_series: bool = True
    log_global_mean_norm_time_series: bool = True
    monthly_reference_data: str | None = None
    time_mean_reference_data: str | None = None

    def build(
        self,
        dataset_info: CoupledDatasetInfo,
        initial_time: xr.DataArray,
        n_timesteps_ocean: int | None = None,
        n_timesteps_ice: int | None = None,
        n_timesteps_atmosphere: int | None = None,
        ocean_normalize: Callable[[TensorMapping], TensorDict] | None = None,
        ice_normalize: Callable[[TensorMapping], TensorDict] | None = None,
        atmosphere_normalize: Callable[[TensorMapping], TensorDict] | None = None,
        save_diagnostics: bool = True,
        output_dir: str | None = None,
        ocean_channel_mean_names: Sequence[str] | None = None,
        ice_channel_mean_names: Sequence[str] | None = None,
        atmosphere_channel_mean_names: Sequence[str] | None = None,
    ) -> "InferenceEvaluatorAggregator":
        if self.monthly_reference_data is None:
            monthly_reference_data = None
        else:
            monthly_reference_data = xr.open_dataset(
                self.monthly_reference_data, decode_timedelta=False
            )
        if self.time_mean_reference_data is None:
            time_mean = None
        else:
            time_mean = xr.open_dataset(
                self.time_mean_reference_data, decode_timedelta=False
            )
        if n_timesteps_atmosphere is not None:
            if n_timesteps_atmosphere > 2**15 and self.log_zonal_mean_images:
                # matplotlib raises an error if image size is too large, and we plot
                # one pixel per timestep in the zonal mean images.
                warnings.warn(
                    "Disabling zonal mean images logging due to large number of timesteps"
                    f" (total atmosphere steps is {n_timesteps_atmosphere}). Set "
                    "log_zonal_mean_images=False or "
                    "decrease n_coupled_steps to avoid this warning."
                )
                log_zonal_mean_images = False
            else:
                log_zonal_mean_images = self.log_zonal_mean_images

        return InferenceEvaluatorAggregator(
            dataset_info=dataset_info,
            n_timesteps_ocean=n_timesteps_ocean,
            n_timesteps_ice=n_timesteps_ice,
            n_timesteps_atmosphere=n_timesteps_atmosphere,
            initial_time=initial_time,
            save_diagnostics=save_diagnostics,
            output_dir=output_dir,
            log_histograms=self.log_histograms,
            log_video=self.log_video,
            enable_extended_videos=self.log_extended_video,
            log_zonal_mean_images=log_zonal_mean_images,
            log_seasonal_means=self.log_seasonal_means,
            log_global_mean_time_series=self.log_global_mean_time_series,
            log_global_mean_norm_time_series=self.log_global_mean_norm_time_series,
            monthly_reference_data=monthly_reference_data,
            time_mean_reference_data=time_mean,
            ocean_normalize=ocean_normalize,
            ice_normalize=ice_normalize,
            atmosphere_normalize=atmosphere_normalize,
            ocean_channel_mean_names=ocean_channel_mean_names,
            ice_channel_mean_names=ice_channel_mean_names,
            atmosphere_channel_mean_names=atmosphere_channel_mean_names,
        )


def _combine_logs(
    n_fast_steps_per_slow_step: int,
    step_key="mean/forecast_step",
    ocean_logs: InferenceLogs | None = None,
    ice_logs: InferenceLogs | None = None,
    atmos_logs: InferenceLogs | None = None,
) -> InferenceLogs:
    """
    Combines ocean, ice, and atmosphere logs into a single list of logs such
    that the ocean logs are aligned to the atmosphere or ice's faster timestep, e.g.

    _combine_logs(
        ocean_logs=[
            {"mean/forecast_step": 0, "o": 0},
            {"mean/forecast_step": 1, "o": 1},
        ],
        atmos_logs=[
            {"mean/forecast_step": 0, "a": 0},
            {"mean/forecast_step": 1, "a": 1},
            {"mean/forecast_step": 2, "a": 2},
            {"mean/forecast_step": 3, "a": 3},
        ],
        n_atmos_steps_per_ocean_step=2,
    )

    results in

    [
        {"mean/forecast_step": 0, "o": 0, "a": 0},
        {"mean/forecast_step": 1, "a": 1},
        {"mean/forecast_step": 2, "o": 1, "a": 2},
        {"mean/forecast_step": 3, "a": 3},
    ]

    """
    combined_logs = []
    if atmos_logs is None:
        for i, log in enumerate(ice_logs):
            if step_key in log and log[step_key] % n_fast_steps_per_slow_step == 0:
                # this is an ocean output step
                ocean_log = ocean_logs[i // n_fast_steps_per_slow_step]
                # sanity check
                assert (
                    ocean_log[step_key] * n_fast_steps_per_slow_step == log[step_key]
                ), (
                    f"Ice forecast step ({log[step_key]}) is not a "
                    f"multiple of the ocean step ({ocean_log[step_key]})."
                )

                combined_logs.append({**ocean_log, **log})
            else:
                # this is an ice-only output step
                combined_logs.append(log)
    elif ice_logs is None:
        for i, log in enumerate(atmos_logs):
            if step_key in log and log[step_key] % n_fast_steps_per_slow_step == 0:
                # this is an ocean output step
                ocean_log = ocean_logs[i // n_fast_steps_per_slow_step]
                # sanity check
                assert (
                    ocean_log[step_key] * n_fast_steps_per_slow_step == log[step_key]
                ), (
                    f"Atmosphere forecast step ({log[step_key]}) is not a "
                    f"multiple of the ocean step ({ocean_log[step_key]})."
                )

                combined_logs.append({**ocean_log, **log})
            else:
                # this is an atmosphere-only output step
                combined_logs.append(log)
    elif ocean_logs is None:
        for i, log in enumerate(atmos_logs):
            if step_key in log and log[step_key] % n_fast_steps_per_slow_step == 0:
                # this is an ice output step
                ice_log = ice_logs[i // n_fast_steps_per_slow_step]
                # sanity check
                assert (
                    ice_log[step_key] * n_fast_steps_per_slow_step == log[step_key]
                ), (
                    f"Atmosphere forecast step ({log[step_key]}) is not a "
                    f"multiple of the ice step ({ice_log[step_key]})."
                )

                combined_logs.append({**ice_log, **log})
            else:
                # this is an atmosphere-only output step
                combined_logs.append(log)
    else:
        for i, log in enumerate(atmos_logs):
            if step_key in log and log[step_key] % n_fast_steps_per_slow_step == 0:
                # this is an ocean output step
                ocean_log = ocean_logs[i // n_fast_steps_per_slow_step]
                # sanity check
                assert (
                    ocean_log[step_key] * n_fast_steps_per_slow_step == log[step_key]
                ), (
                    f"Atmosphere forecast step ({log[step_key]}) is not a "
                    f"multiple of the ocean step ({ocean_log[step_key]})."
                )

                combined_logs.append({**ocean_log, **log})
            else:
                # this is an atmosphere-only output step
                combined_logs.append(ice_logs[i])
                combined_logs.append(log)
    return combined_logs


class InferenceEvaluatorAggregator(
    InferenceAggregatorABC[
        CoupledPairedData | CoupledPrognosticState,
        CoupledPairedData,
    ]
):
    def __init__(
        self,
        dataset_info: CoupledDatasetInfo,
        n_timesteps_ocean: int,
        n_timesteps_ice: int,
        n_timesteps_atmosphere: int,
        initial_time: xr.DataArray,
        ocean_normalize: Callable[[TensorMapping], TensorDict],
        ice_normalize: Callable[[TensorMapping], TensorDict],
        atmosphere_normalize: Callable[[TensorMapping], TensorDict],
        save_diagnostics: bool = True,
        output_dir: str | None = None,
        log_video: bool = False,
        enable_extended_videos: bool = False,
        log_zonal_mean_images: bool = False,
        log_seasonal_means: bool = False,
        log_global_mean_time_series: bool = True,
        log_global_mean_norm_time_series: bool = True,
        monthly_reference_data: xr.Dataset | None = None,
        log_histograms: bool = False,
        time_mean_reference_data: xr.Dataset | None = None,
        ocean_channel_mean_names: Sequence[str] | None = None,
        ice_channel_mean_names: Sequence[str] | None = None,
        atmosphere_channel_mean_names: Sequence[str] | None = None,
    ):
        self._record_ocean_step_20 = n_timesteps_ocean >= 20
        self._record_atmos_step_20 = n_timesteps_atmosphere >= 20

        self._aggregators = {
            "ocean": InferenceEvaluatorAggregator_(
                dataset_info=dataset_info.ocean,
                n_ic_steps=1,
                n_forward_steps=n_timesteps_ocean - 1,
                initial_time=initial_time,
                log_histograms=log_histograms,
                log_video=log_video,
                enable_extended_videos=enable_extended_videos,
                log_zonal_mean_images=log_zonal_mean_images,
                log_seasonal_means=log_seasonal_means,
                log_global_mean_time_series=log_global_mean_time_series,
                log_global_mean_norm_time_series=log_global_mean_norm_time_series,
                monthly_reference_data=monthly_reference_data,
                time_mean_reference_data=time_mean_reference_data,
                log_step_means=[StepMeanEntry(step=20, name="mean_step_20")]
                if self._record_ocean_step_20
                else [],
                channel_mean_names=ocean_channel_mean_names,
                normalize=ocean_normalize,
                save_diagnostics=save_diagnostics,
                output_dir=(
                    os.path.join(output_dir, "ocean")
                    if output_dir is not None
                    else None
                ),
            ),
            "ice": InferenceEvaluatorAggregator_(
                dataset_info=dataset_info.ice,
                n_timesteps=n_timesteps_ice,
                initial_time=initial_time,
                log_histograms=log_histograms,
                log_video=log_video,
                enable_extended_videos=enable_extended_videos,
                log_zonal_mean_images=log_zonal_mean_images,
                log_seasonal_means=log_seasonal_means,
                log_global_mean_time_series=log_global_mean_time_series,
                log_global_mean_norm_time_series=log_global_mean_norm_time_series,
                monthly_reference_data=monthly_reference_data,
                time_mean_reference_data=time_mean_reference_data,
                record_step_20=self._record_ocean_step_20,
                channel_mean_names=ice_channel_mean_names,
                normalize=ice_normalize,
                save_diagnostics=save_diagnostics,
                output_dir=(
                    os.path.join(output_dir, "ice")
                    if output_dir is not None
                    else None
                ),
            ),
            "atmosphere": InferenceEvaluatorAggregator_(
                dataset_info=dataset_info.atmosphere,
                n_ic_steps=1,
                n_forward_steps=n_timesteps_atmosphere - 1,
                initial_time=initial_time,
                log_histograms=log_histograms,
                log_video=log_video,
                enable_extended_videos=enable_extended_videos,
                log_zonal_mean_images=log_zonal_mean_images,
                log_seasonal_means=log_seasonal_means,
                log_global_mean_time_series=log_global_mean_time_series,
                log_global_mean_norm_time_series=log_global_mean_norm_time_series,
                monthly_reference_data=monthly_reference_data,
                time_mean_reference_data=time_mean_reference_data,
                log_step_means=[StepMeanEntry(step=20, name="mean_step_20")]
                if self._record_atmos_step_20
                else [],
                channel_mean_names=atmosphere_channel_mean_names,
                normalize=atmosphere_normalize,
                save_diagnostics=save_diagnostics,
                output_dir=(
                    os.path.join(output_dir, "atmosphere")
                    if output_dir is not None
                    else None
                ),
                log_nino34_index=False,
            ),
        }
        self._num_channels_ocean: int | None = None
        if ocean_channel_mean_names is not None:
            self._num_channels_ocean = len(ocean_channel_mean_names)
        self._num_channels_ice: int | None = None
        if ice_channel_mean_names is not None:
            self._num_channels_ice = len(ice_channel_mean_names)
        self._num_channels_atmos: int | None = None
        if atmosphere_channel_mean_names is not None:
            self._num_channels_atmos = len(atmosphere_channel_mean_names)

    @property
    def ocean(self) -> InferenceEvaluatorAggregator_:
        return self._aggregators["ocean"]
    
    @property
    def ice(self) -> InferenceEvaluatorAggregator_:
        return self._aggregators["ice"]

    @property
    def atmosphere(self) -> InferenceEvaluatorAggregator_:
        return self._aggregators["atmosphere"]

    @torch.no_grad()
    def record_batch(self, data: CoupledPairedData) -> InferenceLogs:
        if self._num_channels_ocean is None:
            self._num_channels_ocean = len(data.ocean_data.prediction)
        if self._num_channels_ice is None:
            self._num_channels_ocean = len(data.ice_data.prediction)
        if self._num_channels_atmos is None:
            self._num_channels_atmos = len(data.atmosphere_data.prediction)
        ocean_logs = self.ocean.record_batch(data.ocean_data)
        ice_logs = self.ice.record_batch(data.ice_data)
        atmos_logs = self.atmosphere.record_batch(data.atmosphere_data)
        n_times_ocean = data.ocean_data.time["time"].size
        n_times_ice = data.ice_data.time["time"].size
        n_times_atmos = data.atmosphere_data.time["time"].size
        return _combine_logs(ocean_logs, ice_logs, atmos_logs, n_times_atmos // n_times_ocean)

    @torch.no_grad()
    def record_initial_condition(
        self,
        initial_condition: CoupledPairedData | CoupledPrognosticState,
    ) -> InferenceLogs:
        """
        Record the initial condition.

        May only be recorded once, before any calls to record_batch.
        """
        ocean_logs = self.ocean.record_initial_condition(initial_condition.ocean_data)
        ice_logs = self.ice.record_initial_condition(initial_condition.ice_data)
        atmos_logs = self.atmosphere.record_initial_condition(
            initial_condition.atmosphere_data
        )
        # initial condition "steps" must align, so record both
        return _combine_logs(ocean_logs, ice_logs, atmos_logs, n_atmos_steps_per_ocean_step=1)

    @torch.no_grad()
    def get_summary_logs(self) -> InferenceLog:
        if self._num_channels_ocean is None or self._num_channels_atmos is None:
            raise ValueError("No data recorded.")
        ocean_logs = self.ocean.get_summary_logs()
        ice_logs = self.ice.get_summary_logs()
        atmos_logs = self.atmosphere.get_summary_logs()
        prefix = "time_mean_norm/rmse"
        ocean_channel_mean = ocean_logs.pop(f"{prefix}/channel_mean")
        ice_channel_mean = ice_logs.pop(f"{prefix}/channel_mean")
        atmos_channel_mean = atmos_logs.pop(f"{prefix}/channel_mean")
        ocean_logs[f"{prefix}/ocean_channel_mean"] = ocean_channel_mean
        ice_logs[f"{prefix}/ice_channel_mean"] = ice_channel_mean
        atmos_logs[f"{prefix}/atmosphere_channel_mean"] = atmos_channel_mean
        channel_mean = (
            ocean_channel_mean * self._num_channels_ocean
            + ice_channel_mean * self._num_channels_ice
            + atmos_channel_mean * self._num_channels_atmos
        ) / (self._num_channels_ocean + self._num_channels_ice + self._num_channels_atmos)
        prefix = "mean_step_20_norm/weighted_rmse"
        if f"{prefix}/channel_mean" in atmos_logs:
            atmos_logs[f"{prefix}/atmosphere_channel_mean"] = atmos_logs.pop(
                f"{prefix}/channel_mean"
            )
        if f"{prefix}/channel_mean" in ocean_logs:
            ocean_logs[f"{prefix}/ocean_channel_mean"] = ocean_logs.pop(
                f"{prefix}/channel_mean"
            )
        if self._record_ice_step_20:
            ice_logs[f"{prefix}/ice_channel_mean"] = ice_logs.pop(
                f"{prefix}/channel_mean"
            )
        duplicates = set(ocean_logs.keys()) & set(ice_logs.keys()) & set(atmos_logs.keys())
        if len(duplicates) > 0:
            raise ValueError(
                "Duplicate keys found in ocean, ice, and atmosphere "
                f"inference evaluator aggregator logs: {duplicates}."
            )
        return {
            "time_mean_norm/rmse/channel_mean": channel_mean,
            **ocean_logs,
            **ice_logs,
            **atmos_logs,
        }

    @torch.no_grad()
    def flush_diagnostics(self, subdir: str | None = None):
        """
        Flushes diagnostics to netCDF files, in separate directories for ocean, ice
        and atmosphere.
        """
        for aggregator in self._aggregators.values():
            aggregator.flush_diagnostics(subdir)


@dataclasses.dataclass
class InferenceAggregatorConfig:
    """
    Configuration for inference aggregator.

    Parameters:
        log_global_mean_time_series: Whether to log global mean time series metrics.
        atmosphere_time_mean_reference_data: Path to atmosphere reference time means.
        ocean_time_mean_reference_data: Path to ocean reference time means.
        ice_time_mean_reference_data: Path to ice reference time means.
    """

    log_global_mean_time_series: bool = True
    atmosphere_time_mean_reference_data: str | None = None
    ocean_time_mean_reference_data: str | None = None
    ice_time_mean_reference_data: str | None = None

    def build(
        self,
        dataset_info: CoupledDatasetInfo,
        n_timesteps_ocean: int,
        n_timesteps_ice: int,
        n_timesteps_atmosphere: int,
        output_dir: str,
    ) -> "InferenceAggregator":
        if self.atmosphere_time_mean_reference_data is None:
            atmos_time_mean = None
        else:
            atmos_time_mean = xr.open_dataset(
                self.atmosphere_time_mean_reference_data, decode_timedelta=False
            )
        if self.ocean_time_mean_reference_data is None:
            ocean_time_mean = None
        else:
            ocean_time_mean = xr.open_dataset(
                self.ocean_time_mean_reference_data, decode_timedelta=False
            )
        if self.ice_time_mean_reference_data is None:
            ice_time_mean = None
        else:
            ice_time_mean = xr.open_dataset(
                self.ice_time_mean_reference_data, decode_timedelta=False
            )
        return InferenceAggregator(
            dataset_info=dataset_info,
            n_timesteps_ocean=n_timesteps_ocean,
            n_timesteps_ice=n_timesteps_ice,
            n_timesteps_atmosphere=n_timesteps_atmosphere,
            output_dir=output_dir,
            log_global_mean_time_series=self.log_global_mean_time_series,
            atmosphere_time_mean_reference_data=atmos_time_mean,
            ocean_time_mean_reference_data=ocean_time_mean,
            ice_time_mean_reference_data=ice_time_mean,
        )


class InferenceAggregator(
    InferenceAggregatorABC[
        CoupledPrognosticState,
        CoupledPairedData,
    ]
):
    """
    Aggregates statistics on a single timeseries of data.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        dataset_info: CoupledDatasetInfo,
        n_timesteps_ocean: int,
        n_timesteps_ice: int,
        n_timesteps_atmosphere: int,
        save_diagnostics: bool = True,
        output_dir: str | None = None,
        atmosphere_time_mean_reference_data: xr.Dataset | None = None,
        ocean_time_mean_reference_data: xr.Dataset | None = None,
        ice_time_mean_reference_data: xr.Dataset | None = None,
        log_global_mean_time_series: bool = True,
    ):
        """
        Args:
            dataset_info: The coordinates of the dataset.
            n_timesteps_ocean: Number of timesteps for ocean.
            n_timesteps_atmosphere: Number of timesteps for atmosphere.
            save_diagnostics: Whether to save diagnostics.
            output_dir: Directory to save diagnostic output.
            atmosphere_time_mean_reference_data: Reference time means for atmosphere.
            ocean_time_mean_reference_data: Reference time means for ocean.
            ice_time_mean_reference_data: Reference time means for ice.
            log_global_mean_time_series: Whether to log global mean time series metrics.
        """
        if save_diagnostics and output_dir is None:
            raise ValueError("Output directory must be set to save diagnostics")
        self._log_time_series = log_global_mean_time_series
        self._save_diagnostics = save_diagnostics
        self._output_dir = output_dir
        self._aggregators = {
            "ocean": InferenceAggregator_(
                dataset_info=dataset_info.ocean,
                n_timesteps=n_timesteps_ocean,
                save_diagnostics=save_diagnostics,
                output_dir=(
                    os.path.join(output_dir, "ocean")
                    if output_dir is not None
                    else None
                ),
                log_global_mean_time_series=log_global_mean_time_series,
                time_mean_reference_data=ocean_time_mean_reference_data,
            ),
            "ice": InferenceAggregator_(
                dataset_info=dataset_info.ice,
                n_timesteps=n_timesteps_ice,
                save_diagnostics=save_diagnostics,
                output_dir=(
                    os.path.join(output_dir, "ice")
                    if output_dir is not None
                    else None
                ),
                log_global_mean_time_series=log_global_mean_time_series,
                time_mean_reference_data=ocean_time_mean_reference_data,
            ),
            "atmosphere": InferenceAggregator_(
                dataset_info=dataset_info.atmosphere,
                n_timesteps=n_timesteps_atmosphere,
                save_diagnostics=save_diagnostics,
                output_dir=(
                    os.path.join(output_dir, "atmosphere")
                    if output_dir is not None
                    else None
                ),
                log_global_mean_time_series=log_global_mean_time_series,
                time_mean_reference_data=atmosphere_time_mean_reference_data,
            ),
        }

    @property
    def ocean(self) -> InferenceAggregator_:
        return self._aggregators["ocean"]

    @property
    def ice(self) -> InferenceAggregator_:
        return self._aggregators["ice"]

    @property
    def atmosphere(self) -> InferenceAggregator_:
        return self._aggregators["atmosphere"]

    @property
    def log_time_series(self) -> bool:
        return self._log_time_series

    @torch.no_grad()
    def record_batch(self, data: CoupledPairedData) -> InferenceLogs:
        ocean_logs = self.ocean.record_batch(data.ocean_data)
        ice_logs = self.ice.record_batch(data.ice_data)
        atmos_logs = self.atmosphere.record_batch(data.atmosphere_data)
        n_times_ocean = data.ocean_data.time["time"].size
        n_times_ice = data.ice_data.time["time"].size
        n_times_atmos = data.atmosphere_data.time["time"].size
        return _combine_logs(ocean_logs, ice_logs, atmos_logs, n_times_atmos // n_times_ocean)

    @torch.no_grad()
    def record_initial_condition(
        self,
        initial_condition: CoupledPrognosticState,
    ) -> InferenceLogs:
        """
        Record the initial condition.

        May only be recorded once, before any calls to record_batch.
        """
        ocean_logs = self.ocean.record_initial_condition(initial_condition.ocean_data)
        ice_logs = self.ice.record_initial_condition(initial_condition.ice_data)
        atmos_logs = self.atmosphere.record_initial_condition(
            initial_condition.atmosphere_data
        )
        # initial condition "steps" must align, so record both
        return _combine_logs(ocean_logs, ice_logs, atmos_logs, n_atmos_steps_per_ocean_step=1)

    def get_summary_logs(self) -> InferenceLog:
        ocean_logs = self.ocean.get_summary_logs()
        ice_logs = self.ice.get_summary_logs()
        atmos_logs = self.atmosphere.get_summary_logs()
        duplicates = set(ocean_logs.keys()) & set(ice_logs.keys()) & set(atmos_logs.keys())
        if len(duplicates) > 0:
            raise ValueError(
                "Duplicate keys found in ocean, ice, and atmosphere "
                f"inference evaluator aggregator logs: {duplicates}."
            )
        return {
            **ocean_logs,
            **ice_logs,
            **atmos_logs,
        }

    @torch.no_grad()
    def flush_diagnostics(self, subdir: str | None = None):
        """
        Flushes diagnostics to netCDF files, in separate directories for ocean, 
        ice, and atmosphere.
        """
        for aggregator in self._aggregators.values():
            aggregator.flush_diagnostics(subdir)

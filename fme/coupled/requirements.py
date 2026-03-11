import dataclasses
import datetime

from fme.ace.requirements import DataRequirements, PrognosticStateDataRequirements


@dataclasses.dataclass
class CoupledDataRequirements:
    ocean_timestep: datetime.timedelta | None = None
    ocean_requirements: DataRequirements | None = None
    ice_timestep: datetime.timedelta | None = None
    ice_requirements: DataRequirements | None = None
    atmosphere_timestep: datetime.timedelta | None = None
    atmosphere_requirements: DataRequirements | None = None

    def _check_requirements(self,
                            fast_requirements,
                            slow_requirements,
                            fast_timestep,
                            slow_timestep,
                            n_steps_fast):
        # check for misconfigured DataRequirements n_timesteps in the fast component
        # note this only checks initial values, if a schedule is used we will not check
        # errors after milestones.
        slow_n_steps = slow_requirements.n_timesteps_schedule.get_value(0)
        fast_n_steps = (slow_n_steps - 1) * n_steps_fast + 1
        fast_n_timesteps = (
            fast_requirements.n_timesteps_schedule.get_value(0)
        )
        if fast_n_timesteps != fast_n_steps:
            raise ValueError(
                f"Fast dataset timestep is {fast_timestep} and "
                f"slow dataset timestep is {slow_timestep}, "
                f"so we need {n_steps_fast} fast steps for each of the "
                f"{slow_n_steps - 1} slow steps, giving {fast_n_steps} total "
                "timepoints (including IC) per sample, but fast dataset "
                f"was configured to return {fast_n_timesteps} "
                "steps."
            )

    def __post_init__(self):
        if self.atmosphere_timestep is None:
            assert (self.ocean_timestep is not None) & (self.ice_timestep is not None)
            if self.ice_timestep > self.ocean_timestep:
                raise ValueError("Ice timestep must be no larger than the ocean's.")
            n_steps_fast = self.ocean_timestep / self.ice_timestep
            fast_t = self.ice_timestep
            slow_t = self.ocean_timestep
            fast_req = self.ice_requirements
            slow_req = self.ocean_requirements
        elif self.ice_timestep is None:
            assert (self.ocean_timestep is not None) & (self.atmosphere_timestep is not None)
            if self.atmosphere_timestep > self.ocean_timestep:
                raise ValueError("Atmosphere timestep must be no larger than the ocean's.")
            n_steps_fast = self.ocean_timestep / self.atmosphere_timestep
            fast_t = self.atmosphere_timestep
            slow_t = self.ocean_timestep
            fast_req = self.atmosphere_requirements
            slow_req = self.ocean_requirements
        elif self.ocean_timestep is None:
            assert (self.atmosphere_timestep is not None) & (self.ice_timestep is not None)
            if self.atmosphere_timestep > self.ice_timestep:
                raise ValueError("Atmosphere timestep must be no larger than the ice's.")
            n_steps_fast = self.ice_timestep / self.atmosphere_timestep
            fast_t = self.atmosphere_timestep
            slow_t = self.ice_timestep
            fast_req = self.atmosphere_requirements
            slow_req = self.ice_requirements
        else:
            if self.ice_timestep > self.ocean_timestep:
                raise ValueError("Ice timestep must be no larger than the ocean's.")
            if self.atmosphere_timestep > self.ocean_timestep:
                raise ValueError("Atmosphere timestep must be no larger than the ocean's.")
            if self.atmosphere_timestep > self.ice_timestep:
                raise ValueError("Atmosphere timestep must be no larger than the ice's.")
            n_steps_fast = self.ocean_timestep / self.atmosphere_timestep
            fast_t = self.atmosphere_timestep
            slow_t = self.ocean_timestep
            fast_req = self.atmosphere_requirements
            slow_req = self.ocean_requirements
        if n_steps_fast != int(n_steps_fast):
            raise ValueError(
                f"Expected fast timestep {fast_t} to be a "
                f"multiple of slow timestep {slow_t}."
            )
        n_steps_fast = int(n_steps_fast)
        self._check_requirements(fast_req, slow_req, fast_t, slow_t, n_steps_fast)
        self._n_steps_fast = n_steps_fast

    @property
    def n_steps_fast(self) -> int:
        return self._n_steps_fast


@dataclasses.dataclass
class CoupledPrognosticStateDataRequirements:
    """
    The requirements for the model's prognostic state.

    """

    ocean: PrognosticStateDataRequirements | None = None
    ice: PrognosticStateDataRequirements | None = None
    atmosphere: PrognosticStateDataRequirements | None = None

from typing import Optional
import realign
from honeyhive.tracer import HoneyHiveTracer
from typing import Dict, Type, Optional, Any
from abc import ABC, abstractmethod
from enum import Enum, auto


class TracingInterface(ABC):

    def __init__(self, trace_source):
        self.trace_source = trace_source

    @abstractmethod
    def initialize_trace(self, *args: Any, **kwargs: Any) -> None:
        ''' Include instrumentation for explicit tracer initialization. '''
        pass

    @abstractmethod
    def enrich_trace(self, *args: Any, **kwargs: Any) -> None:
        ''' Include instrumentation to enrich trace data. '''
        pass

    @abstractmethod
    def initialize_trace_for_simulation(self, run_context: Any) -> None:
        ''' Include instrumentation for simulation tracing. This function is invoked for auto tracing in simulations.'''
        # self.initialize_trace()
        pass


class HoneyHiveTracing(TracingInterface):
    ''' TracingInterface instrumentation for HoneyHive auto tracing '''

    def initialize_trace(self, session_name):
        try:
            HoneyHiveTracer.init(
                api_key=realign.tracing.honeyhive_key,
                project=realign.tracing.honeyhive_project,
                source=self.trace_source,
                session_name=session_name,
            )

        except Exception as e:
            print(e)
        super().initialize_trace()

    def enrich_trace(self, metadata_dict ):
        HoneyHiveTracer.set_metadata(metadata_dict)

    def initialize_trace_for_simulation(self, run_context):
        self.initialize_trace(f'simulation-run-{run_context.run_id}')


class TracerType(Enum):
    # Add value for every tracer implementation
    HONEYHIVE = auto()

class TracerFactory:
    _tracers: Dict[TracerType, Type[TracingInterface]] = {
        # Add value for every tracer implementation
        TracerType.HONEYHIVE: HoneyHiveTracing,
    }

    @classmethod
    def get_tracer(cls, source: str) -> Optional[TracingInterface]:
        active_tracer = cls._get_active_tracer()
        if active_tracer:
            tracer_class = cls._tracers.get(active_tracer)
            if tracer_class:
                return tracer_class(source)
        return None
    
    @staticmethod
    def _get_active_tracer() -> Optional[TracerType]:
        # This function contains the conditional logic for routing to tracer
        if (hasattr(realign.tracing, 'honeyhive_key') and 
            isinstance(realign.tracing.honeyhive_key, str)):
            if (hasattr(realign.tracing, 'honeyhive_project') and 
                isinstance(realign.tracing.honeyhive_project, str)):
                return TracerType.HONEYHIVE
            else:
                raise RuntimeError("Honeyhive Project not found. Please set 'realign.tracing.honeyhive_project' to initiate Honeyhive Tracer.")
        return None

# Function to be used to retrieve the tracer
def get_tracer(source: str) -> Optional[TracingInterface]:
    return TracerFactory.get_tracer(source)
import os
from typing import Optional
import realign
from honeyhive.tracer import HoneyHiveTracer

class TracingInterface():

    def __init__(self):
        pass

    def initialize_trace(self, *args, **kwargs):
        pass

    def add_trace_metadata(self, *args, **kwargs):
        pass

class HoneyHiveTracing(TracingInterface):

    def __init__(self, trace_source):
        # if not 
        self.source = trace_source

    def initialize_trace(self, session_name):
        try:
            HoneyHiveTracer.init(
                api_key=realign.tracing.honeyhive_key,
                project=os.environ['HH_PROJECT'],
                source=self.source,
                session_name=session_name,
            )

        except Exception as e:
            print(e)
        super().initialize_trace()

    def add_trace_metadata(self, metadata_dict ):
        HoneyHiveTracer.set_metadata(metadata_dict)

    def initalize_trace_for_simulation(self, run_context):
        self.initialize_trace('simulation-run')


    


def get_tracer(source: str) -> Optional[TracingInterface]:
    if hasattr(realign.tracing, 'honeyhive_key') and type(realign.tracing.honeyhive_key) == str:
        return HoneyHiveTracing(source)
    else:
        None
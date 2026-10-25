from comfy.comfy_types.node_typing import IO
from comfy_api.latest import io

anything = io.Custom(IO.ANY)

class hswq_ComboClone(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "hswq_ComboClone",
            category = "everywhere",
            display_name = "hswq_ComboClone",
            description = "The combo on this node will replicate whatever the output is connected to",
            inputs   = [
                io.Combo.Input("combo", options=['connect me to a combo widget']),
            ],
            outputs = [
                anything.Output("comboout"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool:
        return isinstance(kwargs.get('combo', None), str)

    @classmethod
    def execute(cls, combo): # type: ignore
        return io.NodeOutput(combo)

class hswq_SimpleString(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "hswq_SimpleString",
            category = "everywhere/deprecated",
            display_name = "hswq_SimpleString",
            description = "Deprecated - use the core comfy string",
            is_deprecated = True,
            inputs   = [
                io.String.Input("string", default=""),
            ],
            outputs = [
                io.String.Output("stringout"),
            ],
        )

    @classmethod
    def execute(cls, string): # type: ignore
        return io.NodeOutput(string)

class hswq_SeedEverywhere(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "hswq_SeedEverywhere",
            category = "everywhere/deprecated",
            display_name = "hswq_SeedEverywhere",
            description = "Deprecated - should automatically be replaced",
            is_deprecated = True,
            inputs   = [
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
            ],
            outputs = [
                io.Int.Output("int"),
            ],
        )
    
    @classmethod
    def execute(cls, seed): # type: ignore
        return io.NodeOutput(seed)

class hswq_AnythingEverywhere(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "hswq_AnythingEverywhere",
            category = "everywhere",
            display_name = "hswq_AnythingEverywhere",
            inputs   = [
                anything.Input("anything", optional=True),
            ],
            outputs = [ ],
        )
    
    @classmethod
    def execute(cls, **kwargs): 
        return io.NodeOutput()

class hswq_AnythingEverywherePrompts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "hswq_AnythingEverywherePrompts",
            category = "everywhere/deprecated",
            display_name = "hswq_AnythingEverywherePrompts", 
            description = "Deprecated - should automatically be replaced",
            is_deprecated = True,
            inputs   = [
                anything.Input("positive", display_name="+ve", optional=True),
                anything.Input("negative", display_name="-ve", optional=True),
            ],
            outputs = [ ],
        )
    
    @classmethod
    def execute(cls, **kwargs): 
        return io.NodeOutput()

class hswq_AnythingEverywhereTriplet(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "hswq_AnythingEverywhereTriplet",
            category = "everywhere/deprecated",
            display_name = "hswq_AnythingEverywhereTriplet", 
            description = "Deprecated - should automatically be replaced",
            is_deprecated = True,
            inputs   = [
                anything.Input("anything", display_name="anything", optional=True),
                anything.Input("anything2", display_name="anything2", optional=True),
                anything.Input("anything3", display_name="anything3", optional=True),
            ],
            outputs = [ ],
        )
    
    @classmethod
    def execute(cls, **kwargs): 
        return io.NodeOutput()
    
class hswq_AnythingSomewhere(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id  = "hswq_AnythingSomewhere",
            category = "everywhere/deprecated",
            display_name = "hswq_AnythingSomewhere", 
            description = "Deprecated - should automatically be replaced",
            is_deprecated = True,
            inputs   = [
                anything.Input("anything", display_name="anything", optional=True),
                io.String.Input("title_regex", default="", optional=True),
                io.String.Input("input_regex", default="", optional=True),
                io.String.Input("group_regex", default="", optional=True),
            ],
            outputs = [ ],
        )
    
    @classmethod
    def execute(cls, **kwargs): 
        return io.NodeOutput()

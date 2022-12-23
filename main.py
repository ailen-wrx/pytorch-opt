from dsl import refactor_DSL
from ipa import moduleIPA


RULE1 = [
    'BertSelfAttention($c, position_embedding_type = $pet) -> $s => epoi.ops.xformers_attn.BertSelfAttention($c, position_embedding_type = $pet, attn_op_name="cutlass") -> $s'
]

RULE2 = [
    'torch.nn.LayerNorm($chs, eps=$clne) -> $sln; torch.nn.Dropout($chdp) -> $sd => epoi.ops.torchscript_ops.FusedDropoutAddLayerNorm($chs, $chdp, eps=$clne) -> self.dropout_add_layernorm',
    'torch.nn.LayerNorm($hs) -> $hs; torch.nn.LayerNorm($hs + $it) => self.dropout_add_layernorm($hs, $it) -> $hs'
]

MODULE_1 = 'higginface_transformers.src.transformers.models.bert.modeling_bert'

def main():
    R = refactor_DSL(RULE1)
    S = moduleIPA(MODULE_1)

    for rule in R.rules:
        for call_sequence in S.call_sequences:
            rule.patternMatch(call_sequence.call_statements)
    pass


if __name__ == '__main__':
    main()
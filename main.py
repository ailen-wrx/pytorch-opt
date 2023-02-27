import copy

import astunparse

from dsl import refactorHelper
from ipa_iterative import moduleIPA
from call_seq_finder import moduleVisitor


RULE1 = [
    'BertSelfAttention($c, position_embedding_type = $pet) -> $s => epoi.ops.xformers_attn.BertSelfAttention($c, position_embedding_type = $pet, attn_op_name="cutlass") -> $s'
]

RULE2 = [
    'torch.nn.LayerNorm($chs, eps=$clne) -> $sln; torch.nn.Dropout($chdp) -> $sd => epoi.ops.torchscript_ops.FusedDropoutAddLayerNorm($chs, $chdp, eps=$clne) -> self.dropout_add_layernorm'
    , '@torch.nn.Dropout($hs) -> $hs; @torch.nn.LayerNorm($hs + $it) -> $hs => self.dropout_add_layernorm($hs, $it) -> $hs'
]

RULE3 = [
    'torch.nn.functional.softmax($as, dim=-1) -> $ap => xformers.triton.softmax.softmax($as) -> $ap'
]

RULE4 = [
    '@torch.nn.Linear(x$) -> $x; <any_assignment>; @torch.nn.Linear(x$) -> $x => <do_not_refactor>'
]

MODULE_0 = 'toy-model.model'
MODULE_1 = 'higginface_transformers.src.transformers.models.bert.modeling_bert'
MODULE_2 = 'higginface_transformers.src.transformers.models.roberta.modeling_roberta'
MODULE_3 = 'higginface_transformers.src.transformers.models.gpt2.modeling_gpt2'
MODULE_4 = 'higginface_transformers.src.transformers.models.distilbert.modeling_distilbert'
MODULE_5 = 'higginface_transformers.src.transformers.models.t5.modeling_t5'
MODULE_6 = 'higginface_transformers.src.transformers.models.albert.modeling_albert'

def main():
    R = refactorHelper(RULE4)

    modules = [MODULE_1, MODULE_2, MODULE_3, MODULE_4, MODULE_5, MODULE_6]
    modules_0 = MODULE_0

    for module in [modules_0]:
        print("[stat] " + module)
        print("")
        S = moduleVisitor(module)
        S.begin_visit()

        cnt = 0
        for cls_idx in S.call_sequences:
            all_match = 1
            before = []
            after  = []
            for rule in R.rules:
                has_match = 0
                for call_sequence in cls_idx:
                    # call_sequence.print()
                    # print()
                    i, j = rule.patternMatch(call_sequence.call_statements)
                    if i != -1:
                        before.append([copy.copy(call_sequence), i, j])
                        after.append(rule.refactor(call_sequence.call_statements[i: j]))
                        has_match = 1
                        # rule.generate(i, j)
                if not has_match: all_match = 0
            if all_match:
                cnt += 1
                for i in range(len(before)):
                    print('-')
                    before[i][0].print(before[i][1], before[i][2])
                    if after[i] is None:
                        continue
                    print('+')
                    for tree in after[i]:
                        print(astunparse.unparse(tree).strip())
                    print()

        print("[stat] " + str(cnt))

    pass


if __name__ == '__main__':
    main()

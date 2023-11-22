// TODO: figure out how to test

pdl.pattern @rewrite_with_args : benefit(1) {
  %input = operand
  %type = type
  %root = operation "torch.aten.expm1"(%input : !pdl.value) -> (%type : !pdl.type)
  rewrite %root {
    replace %root with (%input : !pdl.value)
  }
}

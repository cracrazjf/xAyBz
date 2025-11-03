from psychai.artificial_dataset.xAyBz import XAYBZ
from psychai.tokenizer.tokenizer import print_tokenizer

xaybz = XAYBZ(random_seed=66)
xaybz.save_as_jsonl("./data")
tokenizer = xaybz.tokenizer
print_tokenizer(tokenizer)
tokenizer.save_pretrained("./tokenizer/xaybz_tokenizer")



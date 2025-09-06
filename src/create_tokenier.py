from psychai.artificial_dataset.xAyBz import XAYBZ

xaybz = XAYBZ(random_seed=66)
xaybz.save_as_jsonl("./data")
tokenizer = xaybz.tokenizer
tokenizer.save_pretrained("./tokenizer/xaybz_tokenizer")



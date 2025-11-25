from psychai.artificial_dataset import XAYBZ
from psychai.language import print_tokenizer

def main():
    # ayb experiment
    # xaybz = XAYBZ(random_seed=66)
    # xaybz.save_as_jsonl("./data", dataset_name="ayb")


    # jordan vs elman experiment
    # exp1
    xaybz = XAYBZ(ab_category_size=3, num_y_categories=2, custom="xnor")
    xaybz.save_as_jsonl("./data", dataset_name="je_exp4")
    tokenizer = xaybz.tokenizer
    # print_tokenizer(tokenizer)
    # tokenizer.save_pretrained("./models/je_2024/je_tokenizer")


if __name__ == "__main__":
    main() 


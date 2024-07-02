from datasets import load_dataset, Dataset

def load_and_preprocess_data(dataset_name, subset_fraction=0.2, train_fraction=0.8):
    dataset = load_dataset(dataset_name, split="train")
    subset_size = int(len(dataset) * subset_fraction)
    dataset = dataset.select(range(subset_size))

    train_size = int(len(dataset) * train_fraction)
    train_dataset = Dataset.from_dict(dataset[:train_size])
    val_dataset = Dataset.from_dict(dataset[train_size:])
    
    return train_dataset, val_dataset
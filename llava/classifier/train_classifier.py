# ============================================


# ============================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import os
import argparse
from .prompt_classifier import PromptClassifier, CATEGORY_MAPPING


class PromptDatasetJSONL(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    self.data.append(item)
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['text']
        category = item['category']
        

        label = CATEGORY_MAPPING.get(category, 0)
        
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_classifier(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Category mapping: {CATEGORY_MAPPING}")
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = PromptClassifier(args.model_name, args.num_classes).to(device)
    

    train_dataset = PromptDatasetJSONL(args.train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = None
    if args.val_data:
        val_dataset = PromptDatasetJSONL(args.val_data, tokenizer)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            num_workers=4
        )
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        

        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            category_correct = {i: 0 for i in range(args.num_classes)}
            category_total = {i: 0 for i in range(args.num_classes)}
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    logits = model(input_ids, attention_mask)
                    _, predicted = torch.max(logits, 1)
                    
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    

                    for label, pred in zip(labels, predicted):
                        label_id = label.item()
                        category_total[label_id] = category_total.get(label_id, 0) + 1
                        if label == pred:
                            category_correct[label_id] = category_correct.get(label_id, 0) + 1
            
            val_acc = 100 * val_correct / val_total
            print(f"Validation Acc: {val_acc:.2f}%")
            

            print("Per-category accuracy:")
            id_to_cat = {v: k for k, v in CATEGORY_MAPPING.items()}
            for cat_id in range(args.num_classes):
                if category_total.get(cat_id, 0) > 0:
                    cat_acc = 100 * category_correct.get(cat_id, 0) / category_total[cat_id]
                    cat_name = id_to_cat.get(cat_id, "Unknown")
                    print(f"  {cat_name}: {cat_acc:.2f}% ({category_correct.get(cat_id, 0)}/{category_total[cat_id]})")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                print(f"✓ Best model saved with accuracy: {best_acc:.2f}%")
        
        scheduler.step()
        print("-" * 80)
    

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    tokenizer.save_pretrained(args.output_dir)
    

    with open(os.path.join(args.output_dir, 'category_mapping.json'), 'w') as f:
        json.dump(CATEGORY_MAPPING, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Models saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train prompt classifier for LLaVA")
    parser.add_argument("--train_data", type=str, required=True, 
                        help="Path to training JSONL file")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation JSONL file")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Pretrained model name")
    parser.add_argument("--num_classes", type=int, default=14,
                        help="Number of task classes")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/classifier",
                        help="Output directory for saving models")
    
    args = parser.parse_args()
    train_classifier(args)
import numpy as np
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import inspect
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time

tokenizer = tiktoken.get_encoding('gpt2')

#Changed to GPT-2 params, GPT3 takes 3.4 seconds for 1 step which is too long.
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = tokenizer.n_vocab
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # out = att @ v
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)

        return out
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        rev_std = 1 /math.sqrt(self.config.n_embd)
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            rev_std += (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=rev_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=rev_std)
        
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        fused_avali = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        used_fused = fused_avali and 'cuda' in device
        print("Used fused:", used_fused)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=used_fused)
        return optimizer

    def forward(self, x , targets=None):
        _, T = x.shape
        tok_emb = self.transformer.wte(x)
        pos_idxs = torch.arange(0, T, device=x.device, dtype=torch.long)
        pos_emb = self.transformer.wpe(pos_idxs)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits
    
    def generate(self, x, num_return_sequences=1, max_length = 32):
        model.eval()
        tokens = tokenizer.encode(x)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to('cuda')
        sample_rng = torch.Generator(device='cuda')
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        return xgen

def get_batch(split, batch_size=32, block_size = 128):
    indxs = torch.randint(0, len(split) - block_size, (batch_size, ))
    x = torch.stack([split[i: i + block_size] for i in indxs])
    y = torch.stack([split[i+1: i + block_size + 1] for i in indxs])
    x, y = x.to('cuda'), y.to('cuda')
    return x, y

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    def __init__(self, data, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}
        # /home/ubuntu/GPT3-Dataset/edu_fineweb10B/GPT-2/edu_fineweb10B
        # edu_fineweb10B
        data_root = '/home/ubuntu/GPT3-Dataset/edu_fineweb10B/GPT-2/edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0 ,f'no shards were found'
        if master_process:
            print(f'found {len(shards)} shards for split {split}')
        
        self.reset()

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T).to("cuda")
        y = (buf[1:]).view(B, T).to("cuda")
        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank    
        return x, y
    
    def num_of_batches(self):
        return len(self.tok_data) // (self.B * self.T)

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

torch.set_float32_matmul_precision('high')
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    assert torch.cuda.is_available(), "we need cuda to be avalible"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'using device: {device}')

model = GPT(GPTConfig(vocab_size=50304))
model.to('cuda')
#model = torch.compile(model)
tokenizer = tiktoken.get_encoding('gpt2')
raw_model = model if not hasattr(model, "module") else model.module

if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])

with open(r'input.txt', 'r') as f:
    text = f.read()

#max_steps: 19073

max_lr = 6e-4 * 2
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

train = text[:(int)(len(text) * 0.95)]
val = text[(int)(len(text) * 0.95):]

#B, T = 32, 1024
B, T = 16, 2048

trainloader = DataLoader(train, B, T, ddp_rank, ddp_world_size, "train")
valloader = DataLoader(train, B, T, ddp_rank, ddp_world_size, "val")

# batch: 524288
total_batch_size = 524288
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisable by B * T"
grad_accum_steps = total_batch_size // (B * T)

#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate=3e-4, device='cuda')
eval = 1
eval_iter = 1000

for i in range(max_steps):
    t0 = time.time()

    if i % eval_iter == 0:
        model.eval()
        valloader.reset()
        with torch.no_grad():
            val_loss_total = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = valloader.next_batch()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss /= val_loss_steps
                val_loss_total += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_total, op=dist.ReduceOp.AVG)
        if master_process:
            print(f'Validation loss: {val_loss_total.item():.4f}')
        
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for microstep in range(grad_accum_steps):
        x, y = trainloader.next_batch()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss /= grad_accum_steps
        total_loss += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (microstep == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = trainloader.B * trainloader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f'Step: {i}, Loss: {total_loss:.6f}, learning rate: {lr:.4e}, dt: {dt}, tokens proccessed: {tokens_per_sec} tks/sec')

if ddp:
    destroy_process_group()

torch.save(model.state_dict(), "test_Model.pt")

checkpoint = torch.load("best_val_model.pt", map_location="cuda:0", weights_only=True)

# Remove 'module._orig_mod.' prefix if it exists
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in checkpoint.items():
    # Strip 'module._orig_mod.' from key names
    new_key = k.replace("module._orig_mod.", "")  # Adjust this if another prefix is present
    new_state_dict[new_key] = v

# Load the cleaned state dict
model = GPT(GPTConfig(vocab_size=50304))
model.load_state_dict(new_state_dict, strict=True)
model.to('cuda')
user =''
while True:
    user = input("Enter: ")
    if user.lower() == "stop":
        break
    num_return_sequences = 1
    out = model.generate(user, num_return_sequences, 100)

    for i in range(num_return_sequences):
        temp = out[i]
        temp = tokenizer.decode(temp.tolist())
        print(temp)
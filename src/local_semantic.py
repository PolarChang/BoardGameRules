"""Offline CPU embeddings with versioned content caches; no remote inference."""
import hashlib
import json
import uuid
import time
from src.database import _connect
from src.local_usage import record, cache_hit, document_work_id

MODELS = {
    'e5-small': ('intfloat/multilingual-e5-small', '614241f622f53c4eeff9890bdc4f31cfecc418b3'),
    'minilm': ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'),
}
ENGINE = 'source-embeddings-v2'


def source_text(block):
    fields = block['fields']
    # Generated page labels and inferred chapter titles must not dominate source meaning.
    return fields.get('原文') or fields.get('內容', '')


class SemanticIndex:
    def __init__(self, name, game_id):
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(4)
        model, revision = MODELS[name]
        self.name, self.game_id = name, game_id
        self.version = f'{ENGINE}:{model}:{revision}'
        self.model = SentenceTransformer(model, revision=revision, device='cpu',
                                         local_files_only=True, trust_remote_code=False)

    def vectors(self, texts, query=False):
        prefix = ('query: ' if query else 'passage: ') if self.name == 'e5-small' else ''
        tokenizer = self.model.tokenizer
        # Split by token windows instead of silently truncating long source regions.
        capacity = self.model.max_seq_length - 16
        windows, owners = [], []
        for owner, text in enumerate(texts):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            for start in range(0, max(1, len(tokens)), capacity):
                window = prefix + tokenizer.decode(tokens[start:start + capacity], skip_special_tokens=True)
                if len(tokenizer.encode(window)) > self.model.max_seq_length:
                    raise ValueError('Embedding window exceeds model capacity')
                windows.append(window); owners.append(owner)
        conn = _connect()
        try:
            cur = conn.cursor()
            cur.execute('CREATE TABLE IF NOT EXISTS local_embedding_cache (version TEXT NOT NULL, digest TEXT NOT NULL, vector TEXT NOT NULL, PRIMARY KEY(version,digest))')
            keys = [hashlib.sha256(w.encode()).hexdigest() for w in windows]
            cached = {}
            for key in dict.fromkeys(keys):
                cur.execute('SELECT vector FROM local_embedding_cache WHERE version=? AND digest=?', (self.version,key))
                row=cur.fetchone()
                if row: cached[key]=json.loads(row['vector'])
            conn.commit()
            if cached: cache_hit(self.game_id,self.version,'embedding-query' if query else 'embedding-source',len(cached))
            missing=list(dict.fromkeys(key for key in keys if key not in cached))
            lookup=dict(zip(keys,windows))
            for start in range(0,len(missing),32):
                batch=missing[start:start+32]; values=[lookup[k] for k in batch]
                identifier=uuid.uuid4().hex
                from src.local_import_budget import reserve,settle
                tokens=sum(len(tokenizer.encode(v)) for v in values)
                work_id=hashlib.sha256(json.dumps([self.version,query,batch]).encode()).hexdigest()
                reservation=reserve(self.game_id,'embedding-query' if query else 'embedding-source',tokens,60,family='cpu',work_id=work_id)
                started=time.monotonic()
                record(identifier,self.game_id,self.version,'embedding-query' if query else 'embedding-source',
                       work_id=document_work_id(self.game_id,work_id))
                try:
                    vectors=self.model.encode(values,batch_size=16,normalize_embeddings=True,show_progress_bar=False).tolist()
                except Exception:
                    record(identifier,self.game_id,self.version,'embedding-query' if query else 'embedding-source',outcome='failed')
                    raise
                settle(reservation,tokens,0,time.monotonic()-started,family='cpu')
                record(identifier,self.game_id,self.version,'embedding-query' if query else 'embedding-source',
                       {'prompt_tokens':tokens,'output_tokens':0})
                for key, vector in zip(batch,vectors):
                    cur.execute('INSERT INTO local_embedding_cache VALUES (?,?,?) ON CONFLICT DO NOTHING',(self.version,key,json.dumps(vector)))
                    cached[key]=vector
                conn.commit()
            return [(owner,cached[key]) for owner,key in zip(owners,keys)]
        finally:
            conn.close()

    def prepare(self, blocks):
        self.blocks=[b for b in blocks if b['kind']=='規則' and b.get('evidence_usable') is not False
                     and (b['fields'].get('原文') or b['fields'].get('內容','')).strip()]
        self.documents=self.vectors([source_text(b) for b in self.blocks])

    def rank(self, question, limit=6):
        queries=self.vectors([question],query=True)
        scores={}
        for owner,vector in self.documents:
            score=max(sum(a*b for a,b in zip(vector,q)) for _,q in queries)
            identifier=self.blocks[owner]['id']
            scores[identifier]=max(scores.get(identifier,-1),score)
        return sorted(scores,key=lambda identifier:scores[identifier],reverse=True)[:limit]

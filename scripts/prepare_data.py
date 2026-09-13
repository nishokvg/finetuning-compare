"""Create one reproducible split for both MLX and Fireworks; never tune on test."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Support general', 'Fileservice', 'O365', 'EOL', 'Software', 'Active Directory', 'Computer-Services']
SYSTEM = ('You are an IT helpdesk ticket routing assistant. Given a support ticket, '
          'respond with exactly one of the following categories: ' + ', '.join(LABELS) + '.')

def messages(text, label=None):
    result = [{'role':'system','content':SYSTEM}, {'role':'user','content':f'Support Ticket: {text}'}]
    if label is not None:
        result.append({'role':'assistant','content':label})
    return result

def main():
    source = ROOT / 'support_tickets.csv'
    df = pd.read_csv(source).rename(columns={'category_truth':'label'})
    assert df[['text','label']].notna().all().all(), 'Null data'
    assert set(df.label) == set(LABELS), 'Unexpected labels'
    assert df.text.str.strip().ne('').all(), 'Empty tickets'
    normalized = df.text.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    assert not normalized.duplicated().any(), 'Exact/normalized duplicates need review'
    df['ticket_id'] = [hashlib.sha256(t.encode()).hexdigest()[:16] for t in df.text]
    assert df.ticket_id.is_unique
    # Exactly reproduce the assignment outer split, then carve validation out of development.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    development, test = train_test_split(df, test_size=.2, stratify=df.label, random_state=42)
    train, valid = train_test_split(development, test_size=.2, stratify=development.label, random_state=43)
    splits = {'train': train.reset_index(drop=True), 'valid':valid.reset_index(drop=True), 'test':test.reset_index(drop=True)}
    split_of = {r.ticket_id: name for name, frame in splits.items() for r in frame.itertuples()}
    x = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5)).fit_transform(df.text)
    similarities = (x @ x.T).tocoo()
    near = []
    for i,j,score in zip(similarities.row, similarities.col, similarities.data):
        if i < j and score >= .90:
            a,b = df.iloc[i],df.iloc[j]
            near.append({'id_a':a.ticket_id,'id_b':b.ticket_id,'split_a':split_of[a.ticket_id],
                         'split_b':split_of[b.ticket_id],'similarity':float(score),
                         'label_a':a.label,'label_b':b.label,'text_a':a.text,'text_b':b.text})
    # Keep the assignment test intact. Drop development examples connected to a
    # higher-priority held-out split, including transitive near-duplicate links.
    priority = {'train':0, 'valid':1, 'test':2}
    groups = {tid:{tid} for tid in df.ticket_id}
    for pair in near:
        joined = groups[pair['id_a']] | groups[pair['id_b']]
        for tid in joined: groups[tid] = joined
    excluded = set()
    for tid, group in groups.items():
        highest = max(priority[split_of[t]] for t in group)
        if priority[split_of[tid]] < highest: excluded.add(tid)
    splits = {name: frame[~frame.ticket_id.isin(excluded)].reset_index(drop=True)
              for name, frame in splits.items()}
    assert all(p['id_a'] in excluded or p['id_b'] in excluded or p['split_a']==p['split_b'] for p in near)
    out = ROOT/'data'; out.mkdir(exist_ok=True)
    report = {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'outer_seed':42,'inner_seed':43,'labels':LABELS,'system_prompt':SYSTEM,
              'near_duplicate_threshold':.90,'near_duplicates':near,
              'excluded_ids':sorted(excluded), 'exclusion_policy':'Remove lower-priority examples in connected near-duplicate groups; test > valid > train.', 'splits':{}}
    for name, frame in splits.items():
        csv_path = out/f'{name}.csv'; frame.to_csv(csv_path,index=False)
        data = ''.join(json.dumps({'messages':messages(r.text,r.label)},ensure_ascii=False)+'\n' for r in frame.itertuples())
        (out/f'{name}.jsonl').write_text(data)
        report['splits'][name] = {'rows':len(frame),'counts':frame.label.value_counts().to_dict(),
                                  'jsonl_sha256':hashlib.sha256(data.encode()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('near_duplicates','system_prompt')},indent=2))
    print('Near duplicate pairs >= .90:',len(near),'; cross-split:',sum(r['split_a']!=r['split_b'] for r in near))
    print('Excluded to prevent cross-split leakage:', len(excluded))

if __name__=='__main__': main()

"""Exact source-table lookups and explicit formula arithmetic; no inferred rules."""
import ast
from decimal import Decimal, InvalidOperation
import re


def compute(expression, values):
    tree=ast.parse(expression,mode='eval')
    if len(list(ast.walk(tree)))>30: raise ValueError('Formula too complex')
    def visit(node):
        if isinstance(node,ast.Expression): return visit(node.body)
        if isinstance(node,ast.Name) and node.id in values: return Decimal(str(values[node.id]))
        if isinstance(node,ast.Constant) and type(node.value) in (int,float): return Decimal(str(node.value))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            return -visit(node.operand) if isinstance(node.op,ast.USub) else visit(node.operand)
        if isinstance(node,ast.BinOp):
            a,b=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add): return a+b
            if isinstance(node.op,ast.Sub): return a-b
            if isinstance(node.op,ast.Mult): return a*b
            if isinstance(node.op,ast.Div): return a/b
        raise ValueError('Unsupported formula')
    result=visit(tree)
    if not result.is_finite() or abs(result)>Decimal('1e12'): raise ValueError('Result out of bounds')
    return format(result.normalize(),'f')


def natural_tools(blocks,question,table_query=None):
    from src.preview_retrieval import contains
    claims=[]; values=[]; selected={}
    for block in blocks:
        if block['kind']!='規則' or block.get('evidence_usable') is False: continue
        fields=block['fields']; text=fields.get('原文',''); lines=text.splitlines()
        if block.get('layout_evidence') and len(lines)>=2:
            header=[v.strip() for v in lines[0].split('|')]
            for line in lines[1:]:
                cells=[v.strip() for v in line.split('|')]
                if len(header)!=len(cells) or len(cells)<2 or not header[0] or not cells[0]: continue
                search=table_query or question
                if not contains(search,header[0].lower()) or not contains(search,cells[0].lower()): continue
                columns=[i for i in range(1,len(header)) if header[i] and contains(search,header[i].lower())]
                if len(columns)!=1: continue
                i=columns[0]
                if not re.fullmatch(r'-?\d+(?:\.\d+)?',cells[i]): continue
                claims.append({'text':f'{header[0]} = {header[i]} 時，{cells[0]} = {cells[i]}。',
                    'evidence':[{'id':block['id'],'quote':lines[0]},{'id':block['id'],'quote':line}]})
                values.append((header[0],header[i],cells[0],cells[i])); selected[block['id']]=block
        for line in lines:
            match=re.fullmatch(r'\s*([A-Za-z][A-Za-z_]*)\s*=\s*([A-Za-z0-9_+*/(). -]+)\s*',line)
            if not match or not contains(question,match[1].lower()): continue
            expression=match[2]
            names=set(re.findall(r'[A-Za-z][A-Za-z_]*',expression)); inputs={}
            for name in names:
                found=re.findall(r'(?<![\w])'+re.escape(name)+r'\s*[=：:]\s*(-?\d{1,6}(?:\.\d{1,4})?)(?![\d.])',question,re.I)
                if len(found)==1: inputs[name]=found[0]
            if len(inputs)!=len(names): continue
            try: value=compute(expression,inputs)
            except (ValueError,SyntaxError,ArithmeticError,InvalidOperation): continue
            supplied='、'.join(f'{name}={value}' for name,value in inputs.items())
            claims.append({'text':f'依來源算式 {match[1]} = {expression}，代入你提供的 {supplied}，算術結果為 {value}。',
                           'evidence':[{'id':block['id'],'quote':line}],'calculation':{'formula':expression,'inputs':inputs,'result':value}})
            selected[block['id']]=block
    if not claims: return None
    conflict=any(a[:3]==b[:3] and a[3]!=b[3] for a in values for b in values)
    return {'status':'conflict' if conflict else 'partial','kind':'source_tool','provider':'deterministic',
        'answer':'\n'.join(c['text'] for c in claims),'claims':claims,'citations':list(selected),'retrieved_ids':list(selected),
        'coverage_gaps':['來源數值互相矛盾，不能合併採用。' if conflict else '僅核對明示表格或算式；未替你推定適用條件、取整規則或其他例外。'],
        'sources':[{'id':i,'source':b['fields'].get('來源',''),'page':b['fields'].get('PDF頁碼','')} for i,b in selected.items()]}

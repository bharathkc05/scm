import re

filepath = r"c:\Users\Acer\Desktop\Pipeline\longitudinal_mi_cohort_extraction.md"

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

formatted_lines = []
in_code_block = False

def is_code_start(line):
    # Heuristics for starting a code block
    code_keywords = ['SELECT ', 'CREATE INDEX ', 'CREATE TABLE ', 'import ', 'from ', 'def ', 'class ', 'xgb.XGBClassifier(', 'sns.', 'plt.']
    for kw in code_keywords:
        if line.strip().startswith(kw):
            return True
    # If the line looks like variable assignment with typical data science libs
    if '=' in line and ('dowhy' in line or 'pd.' in line or 'np.' in line):
        return True
    return False

def is_code_line(line):
    if not line.strip():
        return True
    if line.startswith(' ') or line.startswith('\t'):
        return True
    if line.strip().startswith('#'):
        return True
    # If it ends with punctuation like ) or ,
    if line.strip().endswith(',') or line.strip().endswith('(') or line.strip().endswith(']') or line.strip().endswith('}'):
        return True
    if line.strip().startswith('df') or line.strip().startswith('print') or line.strip().startswith('return'):
        return True
    return False

i = 0
while i < len(lines):
    line = lines[i].rstrip('\n')
    
    # Handle bullet points
    line = line.replace('●', '-')
    line = line.replace('○', '  -')
    line = line.replace('■', '    -')
    
    # Handle headings
    if line.startswith('Final Pipeline'):
        line = '# ' + line
    elif line == 'Project Goal':
        line = '## ' + line
    elif line.startswith('PHASE '):
        # Often broken into two lines
        if i + 1 < len(lines) and not lines[i+1].startswith('A.') and not lines[i+1].startswith('B.') and lines[i+1].strip() and not lines[i+1].startswith('-') and not lines[i+1].startswith('●'):
            line = '## ' + line + ' ' + lines[i+1].strip()
            i += 1
        else:
            line = '## ' + line
    elif re.match(r'^[A-Z]\.[1-9][a-z]?:', line) or re.match(r'^[A-Z]\.[1-9][0-9]?:', line):
        line = '### ' + line
    elif line.strip() in ['Protocol', 'Goal', 'Rationale for Design Choices:', 'Architecture Specification:', 'Training Hyperparameters:']:
        line = '#### ' + line
    
    # Highlight bold labels
    if line.startswith('Goal:'):
        line = line.replace('Goal:', '**Goal:**')
    if line.startswith('Deliverable:'):
        line = line.replace('Deliverable:', '**Deliverable:**')
        
    # Code block heuristics (Very basic)
    # We will just enclose lines starting with 'SELECT ', 'import ', 'from ' in ``` if we can detect the block
    if not in_code_block and is_code_start(line):
        in_code_block = True
        formatted_lines.append('```python') # Defaulting to python syntax highlighting for general code
        formatted_lines.append(line)
    elif in_code_block:
        # Determine if we should end the code block
        # End it if we see a markdown list item, heading, or a line that clearly isn't code (like starts with capitalized word, isn't a comment/string)
        if line.startswith('- ') or line.startswith('  -') or line.startswith('###') or line.startswith('##') or (line.strip() and not is_code_line(line) and not is_code_start(line) and line[0].isalpha() and not line.strip().startswith('#') and not line.startswith(' ') and not line.strip().startswith('else:') and not line.strip().startswith('elif')):
            in_code_block = False
            formatted_lines.append('```')
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    else:
        formatted_lines.append(line)
        
    i += 1

if in_code_block:
    formatted_lines.append('```')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write('\n'.join(formatted_lines))

print("Formatting complete.")

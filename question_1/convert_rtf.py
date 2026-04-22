import re
import sys

def rtf_to_text(rtf_path, output_path):
    with open(rtf_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Special handling for Bengali: RTF often puts a space after \uNNNN
    # We replace \uNNNN followed by an optional space with the character
    content = re.sub(r'\\u(\d+) ?', lambda m: chr(int(m.group(1))), content)
    
    # Remove control words
    content = re.sub(r'\\[a-z0-9]+\-?[0-9]*\s?', ' ', content)
    content = content.replace('{', '').replace('}', '')
    
    lines = content.split('\n')
    cleaned_sentences = []
    
    start_recording = False
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if '#' in line and ('.txt' in line.lower() or 'english' in line.lower() or 'bangla' in line.lower()):
            start_recording = True
            continue
            
        if start_recording:
            line = line.replace('\\', '').strip()
            # Collapse multiple spaces into one
            line = re.sub(r'\s+', ' ', line)
            if line:
                cleaned_sentences.append(line)

    with open(output_path, 'w', encoding='utf-8') as f:
        for s in cleaned_sentences:
            f.write(s + '\n')
            
    print(f"Extracted {len(cleaned_sentences)} lines to {output_path}")

if __name__ == '__main__':
    rtf_to_text(sys.argv[1], sys.argv[2])

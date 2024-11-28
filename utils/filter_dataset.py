import gzip
import itertools
import argparse

# Filters out data containing a specific language
def get_ner_reader(data, filter_lang=None):
    fin = gzip.open(data, 'rt') if data.endswith('.gz') else open(data, 'rt')
    for is_divider, lines in itertools.groupby(fin, _is_divider):
        if is_divider:
            continue
        lines = [line.strip().replace('\u200d', '').replace('\u200c', '').replace('\u200b', '') for line in lines]

        metadata = lines[0].strip() if lines[0].strip().startswith('# id') else None
        fields = [line.split() for line in lines if not line.startswith('# id')]
        fields = [list(field) for field in zip(*fields)]
        
        if not f"domain={filter_lang}" in metadata:
            yield fields, metadata
        
        
def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True

    first_token = line.split()[0]
    if first_token == "-DOCSTART-":  # or line.startswith('# id'):  # pylint: disable=simplifiable-if-statement
        return True

    return False

def get_filtered_data_set(text_file, filter_lang=None):
    lines = []
    lines.append('\n')
    
    for fields, metadata in get_ner_reader(data=text_file, filter_lang=filter_lang):
        if metadata is not None:
            lines.append(metadata)
            lines.append('\n')
            
            for field in list(zip(*fields)):
                lines.append(' '.join(field))
                lines.append('\n')
                
            lines.append('\n\n')
    
    with open("test.txt", 'w') as file:
        file.write("".join(lines))
        print("Wrote to", text_file)
            
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--text_file', type=str, default='train.txt')
    argparser.add_argument('--filter_lang' , type=str, default=None)
    
    args = argparser.parse_args()
    text_file = args.text_file
    filter_lang = args.filter_lang
    
    get_filtered_data_set(text_file, filter_lang)
    
    
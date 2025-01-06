import os

EXCLUDED_DIRS = {'__pycache__', 'cache', 'logs', 'tmp', 'node_modules', '.git', '.idea', '.vscode'}
EXCLUDED_FILES = {'.DS_Store', 'thumbs.db'}

OUTPUT_FILE = 'directory_structure.md'

def format_line(name, level, is_dir=False):
    prefix = '    ' * level  # Indentation for subdirectories
    symbol = '/' if is_dir else ''
    return f"{prefix}- {name}{symbol}\n"

def crawl_directory(path, level=0):
    md_lines = []
    entries = sorted(os.listdir(path))
    
    for entry in entries:
        full_path = os.path.join(path, entry)
        
        # Ignore excluded files and directories
        if entry in EXCLUDED_FILES or any(ex in entry for ex in EXCLUDED_DIRS):
            continue

        if os.path.isdir(full_path):
            # Append directory line
            md_lines.append(format_line(entry, level, is_dir=True))
            # Recursively crawl subdirectories
            md_lines.extend(crawl_directory(full_path, level + 1))
        elif os.path.isfile(full_path):
            # Append file line
            md_lines.append(format_line(entry, level, is_dir=False))

    return md_lines

def generate_markdown_structure():
    cwd = os.getcwd()
    print(f"Crawling directory: {cwd}\n")
    
    md_lines = [f"# Directory Structure\n\nRoot: `{cwd}`\n\n"]
    md_lines.extend(crawl_directory(cwd))

    output_path = os.path.join(cwd, OUTPUT_FILE)
    with open(output_path, 'w', encoding='utf-8') as md_file:
        md_file.writelines(md_lines)

    print(f"Directory structure saved to: {output_path}")

if __name__ == "__main__":
    generate_markdown_structure()

import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Crawl a directory and create markdown with all file contents.')
    parser.add_argument('root_dir', help='Root directory to start crawling')
    parser.add_argument('output_file', help='Output markdown file')
    parser.add_argument('--exclude', nargs='*', default=['.git', '__pycache__', '.idea', '.vscode', 'node_modules', 'venv'],
                        help='Directories to exclude from crawling')
    parser.add_argument('--extensions', nargs='*', default=None,
                        help='Only include files with these extensions (e.g., py js txt)')
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.root_dir)
    output_file = args.output_file
    exclude_dirs = set(args.exclude)
    extensions = set(args.extensions) if args.extensions else None
    
    if extensions:
        extensions = {ext if ext.startswith('.') else '.' + ext for ext in extensions}
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write(f"# Code Repository: {os.path.basename(root_dir)}\n\n")
        
        # Write table of contents
        out_file.write("## Table of Contents\n\n")
        
        # First collect all file paths for TOC
        all_files = []
        for root, dirs, files in os.walk(root_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in sorted(files):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)
                
                # Skip files with unwanted extensions if extensions filter is set
                if extensions:
                    file_ext = os.path.splitext(file)[1]
                    if file_ext not in extensions:
                        continue
                
                all_files.append(rel_path)
        
        # Write TOC entries
        for file_path in all_files:
            # Create a link-friendly ID by replacing problematic characters
            link_id = file_path.replace('/', '-').replace('\\', '-').replace('.', '-').replace(' ', '-').lower()
            out_file.write(f"- [{file_path}](#{link_id})\n")
        
        out_file.write("\n## File Contents\n\n")
        
        # Write file contents
        for file_path in all_files:
            full_path = os.path.join(root_dir, file_path)
            # Create a link-friendly ID by replacing problematic characters
            link_id = file_path.replace('/', '-').replace('\\', '-').replace('.', '-').replace(' ', '-').lower()
            
            try:
                # Determine language for syntax highlighting
                file_ext = os.path.splitext(file_path)[1][1:]  # Remove the dot
                
                out_file.write(f"### <a id='{link_id}'></a>{file_path}\n\n")
                out_file.write("```" + file_ext + "\n")
                
                # Read and write file content
                with open(full_path, 'r', encoding='utf-8') as in_file:
                    content = in_file.read()
                    out_file.write(content)
                    # Ensure there's a newline at the end
                    if content and not content.endswith('\n'):
                        out_file.write('\n')
                
                out_file.write("```\n\n")
                out_file.write(f"**End of file: {file_path}**\n\n")
                out_file.write("---\n\n")
            
            except UnicodeDecodeError:
                out_file.write(f"### <a id='{link_id}'></a>{file_path}\n\n")
                out_file.write("*Binary file - content not displayed*\n\n")
                out_file.write(f"**End of file: {file_path}**\n\n")
                out_file.write("---\n\n")
            except Exception as e:
                out_file.write(f"### <a id='{link_id}'></a>{file_path}\n\n")
                out_file.write(f"*Error reading file: {str(e)}*\n\n")
                out_file.write(f"**End of file: {file_path}**\n\n")
                out_file.write("---\n\n")
    
    print(f"Successfully created markdown document: {output_file}")
    print(f"Processed {len(all_files)} files from directory: {root_dir}")

if __name__ == "__main__":
    main()
#!/bin/bash

get_project_root() {
  local dir="${1:-$(pwd)}"
  # Look for a root-level marker. .git is standard, or use your data folder.
  if [[ -d "$dir/.git" ]] || [[ -f "$dir/embryos_data.zip" ]]; then
    echo "$dir"
  elif [[ "$dir" == "/" ]]; then
    exit 1
  else
    get_project_root "$(dirname "$dir")"
  fi
}

PROJECT_DIR=$(get_project_root "$(realpath "$(dirname "$0")")")

SOURCE_DIR="$PROJECT_DIR/data/PGT-A normal abnormal"
NORMAL_DIR="$PROJECT_DIR/data/PGT-A/normal"
ABNORMAL_DIR="$PROJECT_DIR/data/PGT-A/abnormal"

# Create destination folders if they don't exist
mkdir -p "$NORMAL_DIR"
mkdir -p "$ABNORMAL_DIR"

# Iterate through files in source directory
for file in "$SOURCE_DIR"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        
        if [[ "$filename" == *" normal"* ]]; then
            cp "$file" "$NORMAL_DIR/"
            echo "Moved: $filename → normal"
        elif [[ "$filename" == *"abnormal"* ]]; then
            cp "$file" "$ABNORMAL_DIR/"
            echo "Moved: $filename → abnormal"
        fi
    fi
done

echo "Split complete!"
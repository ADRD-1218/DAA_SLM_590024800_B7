#!/bin/bash

echo "================================"
echo "        DAA-SLM Git Push"
echo "================================"

echo ""
echo "Current Git status:"
git status

echo ""
read -p "Enter commit message: " commit_message

if [ -z "$commit_message" ]; then
    echo "Error: Commit message cannot be empty."
    exit 1
fi

echo ""
echo "Adding files..."
git add .

echo ""
echo "Files staged for commit:"
git status --short

echo ""
read -p "Do you want to commit and push these files? [y/N]: " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Push cancelled."
    git restore --staged .
    exit 0
fi

echo ""
echo "Creating commit..."
git commit -m "$commit_message"

if [ $? -ne 0 ]; then
    echo ""
    echo "Commit failed. Push cancelled."
    exit 1
fi

echo ""
echo "Pushing to GitHub..."
git push

if [ $? -ne 0 ]; then
    echo ""
    echo "Push failed."
    exit 1
fi

echo ""
echo "================================"
echo "        Push successful!"
echo "================================"

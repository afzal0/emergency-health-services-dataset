#\!/bin/bash

echo "GitHub Repository Setup for Emergency Health Services Dataset"
echo "==========================================================="
echo ""
echo "Since GitHub CLI is not installed, please:"
echo ""
echo "1. Open your browser and go to: https://github.com/new"
echo ""
echo "2. Create a new repository with these settings:"
echo "   - Repository name: emergency-health-services-dataset"
echo "   - Description: Dataset and analysis code for emergency health services accessibility in Melbourne"
echo "   - Public repository"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo ""
echo "3. After creating, copy your GitHub username and press Enter:"
read -p "Enter your GitHub username: " username
echo ""
echo "Setting up remote..."
git remote remove origin 2>/dev/null
git remote add origin "https://github.com/${username}/emergency-health-services-dataset.git"
echo ""
echo "Pushing to GitHub..."
git push -u origin main
echo ""
echo "Done\! Your repository should now be available at:"
echo "https://github.com/${username}/emergency-health-services-dataset"

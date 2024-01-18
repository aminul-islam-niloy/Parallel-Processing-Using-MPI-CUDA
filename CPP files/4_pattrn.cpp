#include <iostream>
#include <string>

using namespace std;

// Function to count occurrences of a pattern in a text
int countPatternOccurrences(const string& text, const string& pattern) {
    int count = 0;
    size_t pos = text.find(pattern);

    while (pos != string::npos) {
        count++;
        pos = text.find(pattern, pos + pattern.length());
    }

    return count;
}

int main() {
    string paragraph;
    string pattern;

    // Get user input for the paragraph and pattern
    cout << "Enter the paragraph: ";
    getline(cin, paragraph);

    cout << "Enter the pattern: ";
    getline(cin, pattern);

    // Count occurrences of the pattern in the paragraph
    int occurrences = countPatternOccurrences(paragraph, pattern);

    // Display the result
    cout << "Number of occurrences of \"" << pattern << "\": " << occurrences << endl;

    return 0;
}


// Enter the paragraph: This is a sample paragraph containing %x% and %x% occurrences of the %x% pattern.
// Enter the pattern: %x%

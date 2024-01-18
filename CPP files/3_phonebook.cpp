#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;  // Using namespace std

struct Contact {
    string name;
    string phoneNumber;
};

// Function to read contacts from a file
vector<Contact> readPhonebook(const string& filename) {
    vector<Contact> phonebook;
    ifstream inputFile(filename);

    if (!inputFile.is_open()) {
        cerr << "Error opening the file." << endl;
        return phonebook;
    }

    string line;
    while (getline(inputFile, line)) {
        Contact contact;
        size_t pos = line.find(',');
        if (pos != string::npos) {
            contact.name = line.substr(0, pos);
            contact.phoneNumber = line.substr(pos + 1);
            phonebook.push_back(contact);
        }
    }

    inputFile.close();
    return phonebook;
}

// Function to search for contacts by name
vector<Contact> searchContacts(const vector<Contact>&
            phonebook, const string& searchName) {
    vector<Contact> matchingContacts;
    for (const auto& contact : phonebook) {
        if (contact.name.find(searchName) != string::npos) {
            matchingContacts.push_back(contact);
        }
    }
    return matchingContacts;
}

int main() {
    string filename = "phonebook.txt"; 
    string searchName;

    // Read contacts from the file
    vector<Contact> phonebook = readPhonebook(filename);

    // Get user input for the name to search
    cout << "Enter a name to search for: ";
    getline(cin, searchName);

    // Search for contacts matching the name
    vector<Contact> matchingContacts = searchContacts(phonebook, searchName);

    // Display the matching contacts
    if (matchingContacts.empty()) {
        cout << "No contacts found for the given name." << endl;
    } else {
        cout << "Matching contacts:" << endl;
        for (const auto& contact : matchingContacts) {
            cout << "Name: " << contact.name << "\tPhone Number: " << contact.phoneNumber << endl;
        }
    }

    return 0;
}

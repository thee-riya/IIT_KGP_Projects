#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>
#include <chrono>
#include <iomanip>

using namespace std;

// Structure to hold file metadata
// File content, size, creation date and last modified date
struct FileMetadata
{
    string content;
    size_t size;
    string creationDate;
    string lastModified;
};

class MemFS
{
    unordered_map<string, FileMetadata> files;
    mutex fsMutex;               // Mutex to ensure thread safety for shared resources
    atomic<int> createdCount{0}; // Atomic counter for successful file creations

    //  This function returns the current date in the format "dd/mm/yyyy".
    string getCurrentDate() const
    {
        time_t now = time(0);
        tm *ltm = localtime(&now);
        stringstream date;
        date << setw(2) << setfill('0') << ltm->tm_mday << "/"
             << setw(2) << setfill('0') << (ltm->tm_mon + 1) << "/"
             << (1900 + ltm->tm_year);
        return date.str();
    }

    // Helper function to safely create a file (to be used in threads)
    void createFile(const string &filename)
    {
        unique_lock<mutex> lock(fsMutex); // Ensure thread safety when modifying shared resources

        // Check if the file already exists
        if (files.find(filename) != files.end())
        {
            cout << "error: another file with same name exists" << endl;
            return;
        }

        // Initialize file metadata (content, size, creation and modification dates)
        FileMetadata metadata;
        metadata.content = "";
        metadata.size = 0;
        metadata.creationDate = getCurrentDate();
        metadata.lastModified = metadata.creationDate;

        files[filename] = metadata;

        // Increment counter for successful file creation
        this->createdCount++;
    }

    // Helper function to safely write to a file (to be used in threads)
    void writeFile(const string &filename, const string &data)
    {
        unique_lock<mutex> lock(fsMutex);
        if (files.find(filename) == files.end())
            return;

        files[filename].content = data;
        files[filename].size = data.size();
        files[filename].lastModified = getCurrentDate();
    }

    // Helper function to safely delete a file (to be used in threads)
    void deleteFile(const string &filename)
    {
        unique_lock<mutex> lock(fsMutex);
        auto it = files.find(filename);
        if (it == files.end())
        {
            cerr << "Error: " << filename << " doesn’t exist" << endl;
            return;
        }

        files.erase(it);
        cout << "File deleted successfully" << endl;
    }

public:
    // This function returns a single string that is created by concatenating all the strings
    // in the input vector `vec`, separated by the specified `delimiter`.
    string join(const vector<string> &vec, const string &delimiter)
    {
        stringstream ss;
        for (size_t i = 0; i < vec.size(); ++i)
        {
            ss << vec[i];
            if (i != vec.size() - 1)
                ss << delimiter;
        }
        return ss.str();
    }
    // Method to create multiple files
    void createFiles(int numFiles, const vector<string> &filenames)
    {
        vector<thread> threads;

        for (const auto &filename : filenames)
            threads.push_back(thread(&MemFS::createFile, this, filename));

        for (auto &t : threads)
            t.join();

        // If at least one file was created successfully, print success message
        if (this->createdCount.load() > 0)
        {
            if (numFiles == 1)
                cout << "File created successfully" << endl;
            else
                cout << "Files created successfully" << endl;
        }

        // Reset the atomic counter for next command
        this->createdCount.store(0);
    }

    void writeToFile(int numFiles, const vector<pair<string, string>> &fileData)
    {
        vector<thread> threads;

        for (const auto &[filename, data] : fileData)
            threads.push_back(thread(&MemFS::writeFile, this, filename, data));

        for (auto &t : threads)
            t.join();

        // Check if the file is actually present in the files map
        int actualNoOfFiles = 0;
        for (const auto &[filename, _] : fileData)
        {
            if (files.find(filename) == files.end())
                cout << "error: " << filename << " does not exist" << endl;
            else
                actualNoOfFiles++;
        }
        if (actualNoOfFiles > 0)
        {
            if (actualNoOfFiles == 1)
                cout << "successfully written to " << fileData[0].first << endl;
            else
                cout << "successfully written to the given files" << endl;
        }
    }

    void deleteFiles(int numFiles, const vector<string> &filenames)
    {
        vector<thread> threads;
        vector<string> deletedFiles;
        vector<string> nonExistentFiles;

        // Create threads to delete files concurrently
        for (const auto &filename : filenames)
        {
            threads.push_back(thread([this, &deletedFiles, &nonExistentFiles, filename]()
            {
                unique_lock<mutex> lock(fsMutex);
                auto it = files.find(filename);
                // Delete the file if it exists
                if (it != files.end()) {
                    files.erase(it);
                    deletedFiles.push_back(filename);  // Add to deleted files list
                }
                else {
                    nonExistentFiles.push_back(filename); // Add to non-existent files list
                }
            }));
        }

        for (auto &t : threads)
            t.join();

        if (nonExistentFiles.empty() && !deletedFiles.empty())
        {
            if (numFiles == 1)
                cout << "File deleted successfully" << endl;
            else
                cout << "Files deleted successfully" << endl;
        }
        else if (!nonExistentFiles.empty() && deletedFiles.empty())
        {
            // If all files are non-existent and none are deleted
            cout << "File " << join(nonExistentFiles, ", ") << " doesn’t exist" << endl;
        }
        else
        {
            // If some files were deleted and some didn’t exist
            cout << "File " << join(nonExistentFiles, ", ") << " doesn’t exist and remaining files deleted successfully" << endl;
        }
    }

    void readFile(const string &filename) const
    {
        auto it = files.find(filename);
        if (it != files.end())
            cout << it->second.content << endl;
        else
            cerr << "Error: " << filename << " does not exist" << endl;
    }

    void listFiles(bool detailed = false) const
    {
        if (detailed)
        {
            cout << "size created last modified filename" << endl;
            for (const auto &[filename, metadata] : files)
                cout << metadata.size << " " << metadata.creationDate << " " << metadata.lastModified << " " << filename << endl;
        }
        else
            for (const auto &[filename, _] : files)
                cout << filename << endl;
    }

    void executeCommand(const string &command)
    {
        istringstream iss(command);
        string token;
        iss >> token;

        if (token == "create")
        {
            int numFiles = 1;
            vector<string> filenames;

            if (iss >> token && token == "-n")
                iss >> numFiles;
            else
                filenames.push_back(token);

            while (iss >> token)
                filenames.push_back(token);

            if (filenames.size() != static_cast<size_t>(numFiles))
            {
                cerr << "error: number of filenames does not match specified -n value" << endl;
                return;
            }

            createFiles(numFiles, filenames);
        }
        else if (token == "write")
        {
            int numFiles = 1;
            vector<pair<string, string>> fileData;

            if (iss >> token && token == "-n")
                iss >> numFiles;
            else
            {
                string filename = token;
                string data;
                iss >> quoted(data);
                fileData.push_back({filename, data});
            }

            while (iss >> token)
            {
                string filename = token;
                string data;
                if (!(iss >> quoted(data)))
                {
                    cerr << "error: missing text for " << filename << endl;
                    return;
                }
                fileData.push_back({filename, data});
            }

            if (fileData.size() != static_cast<size_t>(numFiles))
            {
                cerr << "error: number of file data pairs does not match specified -n value" << endl;
                return;
            }

            writeToFile(numFiles, fileData);
        }
        else if (token == "delete")
        {
            int numFiles = 1;
            vector<string> filenames;

            if (iss >> token && token == "-n")
                iss >> numFiles;
            else
                filenames.push_back(token);

            while (iss >> token)
                filenames.push_back(token);

            if (filenames.size() != static_cast<size_t>(numFiles))
            {
                cerr << "error: number of filenames does not match specified -n value" << endl;
                return;
            }

            deleteFiles(numFiles, filenames);
        }
        else if (token == "read")
        {
            if (iss >> token)
                readFile(token);
            else
                cerr << "error: no filename provided for read command" << endl;
        }
        else if (token == "ls")
        {
            bool detailed = false;
            if (iss >> token && token == "-l")
                detailed = true;
            listFiles(detailed);
        }
        else if (token == "exit")
        {
            cout << "exiting memFS" << endl;
            exit(0);
        }
        else
            cerr << "error: invalid command" << endl;
    }
};

int main()
{
    MemFS memFS;
    string command;

    while (true)
    {
        cout << "memfs> ";
        getline(cin, command);
        memFS.executeCommand(command);
    }
    return 0;
}

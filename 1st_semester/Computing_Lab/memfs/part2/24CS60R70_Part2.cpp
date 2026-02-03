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
#include <sys/resource.h>

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
private:
    unordered_map<string, FileMetadata> files;
    mutex fsMutex;
    atomic<int> createdCount{0};

    // Format current date
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

    // File creation helper
    void createFile(const string &filename)
    {
        unique_lock<mutex> lock(fsMutex);
        if (files.find(filename) != files.end())
        {
            cerr << "error: file with same name exists" << endl;
            return;
        }
        FileMetadata metadata = {"", 0, getCurrentDate(), getCurrentDate()};
        files[filename] = metadata;
        createdCount++;
    }

    // File writing helper
    void writeFile(const string &filename, const string &data)
    {
        unique_lock<mutex> lock(fsMutex);
        if (files.find(filename) == files.end())
        {
            cerr << "Error: " << filename << " does not exist" << endl;
            return;
        }
        files[filename].content = data;
        files[filename].size = data.size();
        files[filename].lastModified = getCurrentDate();
    }

    // File reading helper
    void readFile(const string &filename)
    {
        unique_lock<mutex> lock(fsMutex);
        if (files.find(filename) == files.end())
        {
            cerr << "Error: " << filename << " does not exist" << endl;
            return;
        }
        string content = files[filename].content; // Just accessing the content to simulate a read
    }

    // File deletion helper
    void deleteFile(const string &filename)
    {
        unique_lock<mutex> lock(fsMutex);
        if (files.erase(filename) == 0)
            cerr << "Error: " << filename << " doesn’t exist" << endl;
    }

public:
    void createFiles(int numFiles, const vector<string> &filenames)
    {
        vector<thread> threads;
        for (const auto &filename : filenames)
            threads.push_back(thread(&MemFS::createFile, this, filename));
        for (auto &t : threads)
            t.join();
        createdCount = 0;
    }

    void writeToFile(int numFiles, const vector<pair<string, string>> &fileData)
    {
        vector<thread> threads;
        for (const auto &[filename, data] : fileData)
            threads.push_back(thread(&MemFS::writeFile, this, filename, data));
        for (auto &t : threads)
            t.join();
    }

    void readFromFile(int numFiles, const vector<string> &filenames)
    {
        vector<thread> threads;
        for (const auto &filename : filenames)
            threads.push_back(thread(&MemFS::readFile, this, filename));
        for (auto &t : threads)
            t.join();
    }

    void deleteFiles(int numFiles, const vector<string> &filenames)
    {
        vector<thread> threads;
        for (const auto &filename : filenames)
            threads.push_back(thread(&MemFS::deleteFile, this, filename));
        for (auto &t : threads)
            t.join();
    }
};

// Benchmarking the MemFS
class MemFSBenchmark
{
public:
    void benchmark(MemFS &fs, int numFiles, int numThreads)
    {
        vector<string> filenames;
        vector<pair<string, string>> fileData;
        const int batchSize = 1000; // Process files in batches of 1000 to control resource use

        // Prepare filenames and file data
        for (int i = 0; i < numFiles; ++i)
        {
            filenames.push_back("file" + to_string(i));
            fileData.push_back({"file" + to_string(i), "Sample content"});
        }

        auto start = chrono::high_resolution_clock::now();

        // Process in batches to control resource usage
        for (int batchStart = 0; batchStart < numFiles; batchStart += batchSize)
        {
            int currentBatchSize = min(batchSize, numFiles - batchStart);
            vector<thread> threads;

            // Creating files in the current batch
            for (int i = 0; i < numThreads; ++i)
            {
                int startIdx = batchStart + i * (currentBatchSize / numThreads);
                int endIdx = min(startIdx + (currentBatchSize / numThreads), batchStart + currentBatchSize);

                if (startIdx < endIdx)
                {
                    threads.push_back(thread(&MemFS::createFiles, &fs, endIdx - startIdx,
                                             vector<string>(filenames.begin() + startIdx, filenames.begin() + endIdx)));
                }
            }
            for (auto &t : threads)
                t.join();
            threads.clear();

            // Writing to files in the current batch
            for (int i = 0; i < numThreads; ++i)
            {
                int startIdx = batchStart + i * (currentBatchSize / numThreads);
                int endIdx = min(startIdx + (currentBatchSize / numThreads), batchStart + currentBatchSize);

                if (startIdx < endIdx)
                {
                    threads.push_back(thread(&MemFS::writeToFile, &fs, endIdx - startIdx,
                                             vector<pair<string, string>>(fileData.begin() + startIdx, fileData.begin() + endIdx)));
                }
            }
            for (auto &t : threads)
                t.join();
            threads.clear();

            // Reading files in the current batch
            for (int i = 0; i < numThreads; ++i)
            {
                int startIdx = batchStart + i * (currentBatchSize / numThreads);
                int endIdx = min(startIdx + (currentBatchSize / numThreads), batchStart + currentBatchSize);

                if (startIdx < endIdx)
                {
                    threads.push_back(thread(&MemFS::readFromFile, &fs, endIdx - startIdx,
                                             vector<string>(filenames.begin() + startIdx, filenames.begin() + endIdx)));
                }
            }
            for (auto &t : threads)
                t.join();
            threads.clear();

            // Deleting files in the current batch
            for (int i = 0; i < numThreads; ++i)
            {
                int startIdx = batchStart + i * (currentBatchSize / numThreads);
                int endIdx = min(startIdx + (currentBatchSize / numThreads), batchStart + currentBatchSize);

                if (startIdx < endIdx)
                {
                    threads.push_back(thread(&MemFS::deleteFiles, &fs, endIdx - startIdx,
                                             vector<string>(filenames.begin() + startIdx, filenames.begin() + endIdx)));
                }
            }
            for (auto &t : threads)
                t.join();
        }

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count(); // in milliseconds
        double avgLatency = duration / static_cast<double>(numFiles);

        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        cout << "Number of Threads: " << numThreads << endl;
        cout << "Average Latency: " << avgLatency << " milliseconds" << endl;
        cout << "CPU Usage: " << usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6 << " seconds" << endl;
        cout << "Memory Usage: " << usage.ru_maxrss << " KB" << endl;
    }
};

int main()
{
    MemFS fs;
    MemFSBenchmark benchmark;

    cout << "Benchmarking MemFS:" << endl;

    // Run benchmarks for different numbers of threads and file counts
    for (int threads : {1, 2, 4, 8, 16})
    {
        cout << "Benchmark with 100 files, " << threads << " threads" << endl;
        benchmark.benchmark(fs, 100, threads);
        cout << endl;

        cout << "Benchmark with 1000 files, " << threads << " threads" << endl;
        benchmark.benchmark(fs, 1000, threads);
        cout << endl;

        cout << "Benchmark with 10000 files, " << threads << " threads" << endl;
        benchmark.benchmark(fs, 10000, threads);
        cout << endl;
    }

    return 0;
}
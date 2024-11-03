package clique

import (
	"encoding/csv"
	"os"
	"strconv"
)

var csvFileName = "node_data.csv"

// WriteToCSV writes node data to a CSV file
func WriteToCSV(address string, stake uint64, transactionTime string, blockSize uint64, blockCreationTime string, blockHash string) error {
	file, err := os.OpenFile(csvFileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Create a record with the values
	record := []string{address, strconv.FormatUint(stake, 10), transactionTime, strconv.FormatUint(blockSize, 10), blockCreationTime, blockHash}
	if err := writer.Write(record); err != nil {
		return err
	}

	return nil
}


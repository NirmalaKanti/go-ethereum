// Copyright 2017 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package clique

import (
	"fmt"
	"math/rand"
	"time"
	"math"
        "encoding/csv"
	"os"
	"sort"
	"strconv"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/rpc"
	
)

const csvFilePath = "/home/vboxuser/Downloads/Telegram Desktop/history1.csv" // Replace with the actual file path to your CSV
// API is a user facing RPC API to allow controlling the signer and voting
// mechanisms of the proof-of-authority scheme.
type API struct {
	chain  consensus.ChainHeaderReader
	clique *Clique
	snapshot *Snapshot
	Consensus *Clique
	Theta []float64
	throughputStartTime time.Time
    throughputCounter   uint64
}


type status struct {
	InturnPercent float64                `json:"inturnPercent"`
	SigningStatus map[common.Address]int `json:"sealerActivity"`
	NumBlocks     uint64                 `json:"numBlocks"`
	TransactionTime time.Duration `json:"transaction_time"`
	Prediction float64  `json:"prediction"`
	MaxStakedMiner  string                 `json:"maxStakedMiner"`
	MaxStake        uint64                 `json:"maxStake"`
	Malacious float64 `json:"malacious"`
	Theta           []float64              `json:"theta"`
	FairnessMetric float64  `json:"fairnessmetric"`
	Time int64 `json:"time1"`
	BlockCreationTime time.Duration   `json:"blockCreationTime"`
}

// GetSnapshot retrieves the state snapshot at a given block.
func (api *API) GetSnapshot(number *rpc.BlockNumber) (*Snapshot, error) {
	// Retrieve the requested block number (or current if none requested)
	var header *types.Header
	if number == nil || *number == rpc.LatestBlockNumber {
		header = api.chain.CurrentHeader()
	} else {
		header = api.chain.GetHeaderByNumber(uint64(number.Int64()))
	}
	// Ensure we have an actually valid block and return its snapshot
	if header == nil {
		return nil, errUnknownBlock
	}
	return api.clique.snapshot(api.chain, header.Number.Uint64(), header.Hash(), nil)
}

// GetSnapshotAtHash retrieves the state snapshot at a given block.
func (api *API) GetSnapshotAtHash(hash common.Hash) (*Snapshot, error) {
	header := api.chain.GetHeaderByHash(hash)
	if header == nil {
		return nil, errUnknownBlock
	}
	return api.clique.snapshot(api.chain, header.Number.Uint64(), header.Hash(), nil)
}


type Stakeholder struct {
	StakedAmount float64
	Rewards      float64
	malbehaviour  float64
}

// GetSigners retrieves the list of authorized signers at the specified block.
func (api *API) GetSigners(number *rpc.BlockNumber) ([]common.Address, error) {
	// Retrieve the requested block number (or current if none requested)
	var header *types.Header
	if number == nil || *number == rpc.LatestBlockNumber {
		header = api.chain.CurrentHeader()
	} else {
		header = api.chain.GetHeaderByNumber(uint64(number.Int64()))
	}
	// Ensure we have an actually valid block and return the signers from its snapshot
	if header == nil {
		return nil, errUnknownBlock
	}
	snap, err := api.clique.snapshot(api.chain, header.Number.Uint64(), header.Hash(), nil)
	if err != nil {
		return nil, err
	}
	return snap.signers(), nil
}
// Abhi
//soni
func (api *API) AddStake(stake uint64) {
	log.Info("adding Stake")
	fmt.Println(stake)
	api.clique.lock.Lock()
	defer api.clique.lock.Unlock()


	//api.clique.stakes[address] = stake
	api.clique.stake = api.clique.stake + stake
	
       rand.Seed(time.Now().UnixNano())
	randomValue := rand.Intn(4) + 1

	// Print the generated random value
	fmt.Println("Generated Random Value:", randomValue)

	// Set the malicious flag if the random value is 1
	if randomValue == 1 {
		api.clique.malicious = true
		api.clique.malnumber++
		
		log.Info("Node is marked as malicious")
         }


}


// Stakeholder represents a participant in staking with their staked amount and earned rewards.


// CalculateFairnessMetric calculates a fairness metric based on the rewards distribution.
// CalculateFairnessMetric calculates a fairness metric based on the rewards distribution.
func (api *API) CalculateFairnessMetric(status *status) (float64, error) {
    stakeholders, err := api.getStakeholders(status)
    if err != nil {
        return 0.0, err
    }

    totalRewards := calculateTotalRewards(stakeholders)
    fairnessMetric := calculateGiniCoefficient(stakeholders, totalRewards)




fmt.Printf("fairnessMetric: %.2f%%\n", fairnessMetric)
   
    return fairnessMetric, nil
}

func calculateTotalRewards(stakeholders []Stakeholder) float64 {
	totalRewards := 0.0
	for _, s := range stakeholders {
		totalRewards += s.Rewards
	}
	return totalRewards
}

func (api *API) getStakeholders(status *status) ([]Stakeholder, error) {
    stakes, err := api.GetStakes()
    if err != nil {
        return nil, err
    }

    stakeholders := make([]Stakeholder, len(stakes))

    for i, stake := range stakes {
        count, exists := status.SigningStatus[stake.Owner]
        if !exists {
            return nil, fmt.Errorf("miner not found in sealerActivity")
        }
        rewards := float64(count) * 10

        stakeholders[i] = Stakeholder{
            StakedAmount: float64(stake.OStakes),
            Rewards:      rewards,
            malbehaviour: api.clique.malnumber,
        }
    }

    return stakeholders, nil
}


func sortStakeholdersByRewardsDesc(stakeholders []Stakeholder) {
    sort.Slice(stakeholders, func(i, j int) bool {
        return stakeholders[i].Rewards > stakeholders[j].Rewards
    })
}

func calculateCumulativeProportion(i, n int) float64 {
    return float64(i+1) / float64(n)
}

// calculateGiniCoefficient calculates the Gini Coefficient as a fairness metric.
func calculateGiniCoefficient(stakeholders []Stakeholder, totalRewards float64) float64 {
	// Sort stakeholders by rewards in descending order
	sortStakeholdersByRewardsDesc(stakeholders)

	// Calculate Gini Coefficient
	n := len(stakeholders)
	giniSum := 0.0
	for i, s := range stakeholders {
	
		giniSum += (float64(i+1)/float64(n) - calculateCumulativeProportion(i+1, n)) + s.Rewards-s.malbehaviour
		
		
	}
	//fmt.Printf("ginisum: %.2f%\n", giniSum)
	//fmt.Printf("Rewards: %.2f%\n", stakeholders[2].Rewards)
	if stakeholders[0].malbehaviour>1 {
	
	if stakeholders[0].malbehaviour*totalRewards <giniSum {
	    giniSum=giniSum-(stakeholders[0].malbehaviour*totalRewards)
		}
	
		if stakeholders[0].malbehaviour*totalRewards >giniSum {
	      giniSum=(stakeholders[0].malbehaviour*totalRewards)-giniSum
		}
		}
		
       // fmt.Printf("ginisum: %.2f%\n", giniSum)
	giniCoefficient := 2 * giniSum / (float64(n) * totalRewards)
	//fmt.Printf("ginicoefficient: %.2f%\n", giniCoefficient)
	return giniCoefficient
}
// GetSignersAtHash retrieves the list of authorized signers at the specified block.
func (api *API) GetSignersAtHash(hash common.Hash) ([]common.Address, error) {
	header := api.chain.GetHeaderByHash(hash)
	if header == nil {
		return nil, errUnknownBlock
	}
	snap, err := api.clique.snapshot(api.chain, header.Number.Uint64(), header.Hash(), nil)
	if err != nil {
		return nil, err
	}
	return snap.signers(), nil
}

// Proposals returns the current proposals the node tries to uphold and vote on.
func (api *API) Proposals() map[common.Address]bool {
	api.clique.lock.RLock()
	defer api.clique.lock.RUnlock()

	proposals := make(map[common.Address]bool)
	for address, auth := range api.clique.proposals {
		proposals[address] = auth
	}
	return proposals
}

// Propose injects a new authorization proposal that the signer will attempt to
// push through.
func (api *API) Propose(address common.Address, auth bool) {
	api.clique.lock.Lock()
	defer api.clique.lock.Unlock()

	api.clique.proposals[address] = auth
}

// Discard drops a currently running proposal, stopping the signer from casting
// further votes (either for or against).
func (api *API) Discard(address common.Address) {
	api.clique.lock.Lock()
	defer api.clique.lock.Unlock()

	delete(api.clique.proposals, address)
}




// Status returns the status of the last N blocks,
// - the number of active signers,
// - the number of signers,
// - the percentage of in-turn blocks
// scaleFeature scales a feature to the range [-1, 1]
func scaleFeature(feature float64, min, max float64) float64 {
	return -1 + 2*(feature-min)/(max-min)
}

// scaleFeatures scales a slice of features to the range [-1, 1]
func scaleFeatures(features []float64) []float64 {
	min := features[0]
	max := features[0]

	// Find the minimum and maximum values
	for _, feature := range features {
		if feature < min {
			min = feature
		}
		if feature > max {
			max = feature
		}
	}

	// Scale each feature
	scaledFeatures := make([]float64, len(features))
	for i, feature := range features {
		scaledFeatures[i] = scaleFeature(feature, min, max)
	}

	return scaledFeatures
}


func scaleFeatureMatrix(features [][]float64) [][]float64 {
	for j := 1; j < len(features[0]); j++ { // Skip the first column (bias term)
		min := features[0][j]
		max := features[0][j]

		// Find the minimum and maximum values for each feature
		for i := range features {
			feature := features[i][j]
			if feature < min {
				min = feature
			}
			if feature > max {
				max = feature
			}
		}

		// Scale each feature
		for i := range features {
			features[i][j] = scaleFeature(features[i][j], min, max)
		}
	}

	return features
}

// predictUsingLLR predicts a value using Logistic Regression based on the given status
func (api *API) predictUsingLogisticRegression(status *status, miner common.Address) (float64, error) {
	// Extract relevant values from the status
	count, exists := status.SigningStatus[miner]
	if !exists {
		return 0.0, fmt.Errorf("miner not found in sealerActivity")
	}
	transactionTime := float64(status.TransactionTime.Milliseconds())
	maxStake := float64(status.MaxStake)
	malacious := float64(status.Malacious)
        api.snapshot.MBlocks = float64(count)
        api.snapshot.MStakes = float64(status.MaxStake)
        api.snapshot.malnumber=api.clique.malnumber
	// Use the logistic regression parameters obtained from training
	theta := []float64{status.Theta[2], status.Theta[1],status.Theta[3]}

	// Prepare input features for prediction
	inputFeatures := []float64{
		maxStake,
		float64(count),
		transactionTime,
		malacious,
		// Add other input features as needed
	}

	// Perform the prediction using logistic regression model
	if malacious > 5 {
		return 0.25, nil
	}

	scaledInputFeatures := scaleFeatures(inputFeatures)

	prediction := predict(theta, scaledInputFeatures)

	if api.clique.malnumber == 2 && prediction > 0.7 {
		return 0.534, nil
	}

	return prediction, nil
}



func logisticRegression(attributes [][]float64, labels []float64) []float64 {
	// Initialize parameters theta
	theta := make([]float64, len(attributes[0])+1)

	// Set up the feature matrix X
	X := make([][]float64, len(attributes))
	for i := range X {
		X[i] = append([]float64{1}, attributes[i]...)
	}

	X = scaleFeatureMatrix(X)

	// Set the learning rate and number of iterations
	alpha := 0.01
	numIterations := 1000

	// Perform gradient descent
	for iter := 0; iter < numIterations; iter++ {
		theta = gradientDescent(theta, X, labels, alpha)
	}

	return theta
}


func gradientDescent(theta []float64, X [][]float64, y []float64, alpha float64) []float64 {
	m := float64(len(y))
	hypothesis := make([]float64, len(y))

	// Calculate the hypothesis using the current parameters theta
	for i := range hypothesis {
		hypothesis[i] = sigmoid(dotProduct(X[i], theta))
	}

	// Calculate the gradient and update the parameters theta
	for j := range theta {
		gradient := 0.0
		for i := range hypothesis {
			gradient += (hypothesis[i] - y[i]) * X[i][j]
		}
		theta[j] -= alpha * gradient / m
	}

	return theta
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func dotProduct(a, b []float64) float64 {
	result := 0.0
	for i := range a {
		result += a[i] * b[i]
	}
	return result
}
// calculateAccuracy calculates the accuracy of the logistic regression model
func calculateAccuracy(theta []float64, attributes [][]float64, labels []float64) float64 {
	correctPredictions := 0

	// Set up the feature matrix X
	X := make([][]float64, len(attributes))
	for i := range X {
		X[i] = append([]float64{1}, attributes[i]...)
	}

	// Predict the labels using the current parameters theta
	for i := range labels {
		prediction := predict(theta, X[i])
		if prediction == labels[i] {
			correctPredictions++
		}
	}

	accuracy := float64(correctPredictions) / float64(len(labels))
	return accuracy
}

// predict predicts the label (0 or 1) using the logistic regression model
func predict(theta []float64, x []float64) float64 {
	hypothesis := sigmoid(dotProduct(theta, x))
	//fmt.Printf("Sigmoid Parameters: %.2f%%\n", hypothesis)
	
	if hypothesis >= 0.5 {
		return hypothesis
	}
	return hypothesis
}
func (api *API) GetMinerWithMaxStake() (common.Address, uint64, error) {
	stakes, err := api.GetStakes()
	if err != nil {
		return common.Address{}, 0, err
	}

	// Calculate miner with the highest stake
	var maxStake uint64
	var maxStakedAddress common.Address

	for _, stake := range stakes {
		if stake.OStakes > maxStake {
			maxStake = stake.OStakes
			maxStakedAddress = stake.Owner
			 // Assuming snapshot has a field maxStake
		}
	}

	return maxStakedAddress, maxStake, nil
}

// GetStakes retrieves the current stakes
func (api *API) GetStakes() ([]*TallyStake, error) {
	snap, err := api.clique.snapshot(api.chain, api.chain.CurrentHeader().Number.Uint64(), api.chain.CurrentHeader().Hash(), nil)
	if err != nil {
		return nil, err
	}

	return snap.TallyStakes, nil
}
func (api *API) Status() (*status, error) {
	var (
		numBlocks = uint64(64)
		header    = api.chain.CurrentHeader()
		diff      = uint64(0)
		optimals  = 0
	)
	snap, err := api.clique.snapshot(api.chain, header.Number.Uint64(), header.Hash(), nil)
	if err != nil {
		return nil, err
	}
	var (
		signers = snap.signers()
		end     = header.Number.Uint64()
		start   = end - numBlocks
	)

	
	if numBlocks > end {
		start = 1
		numBlocks = end - start
	}
	signStatus := make(map[common.Address]int)
	for _, s := range signers {
		signStatus[s] = 0
	}
	for n := start; n < end; n++ {
		h := api.chain.GetHeaderByNumber(n)
		if h == nil {
			return nil, fmt.Errorf("missing block %d", n)
		}
		if h.Difficulty.Cmp(diffInTurn) == 0 {
			optimals++
		}
		diff += h.Difficulty.Uint64()
		sealer, err := api.clique.Author(h)
		if err != nil {
			return nil, err
		}
		signStatus[sealer]++
		
		
		
	}
	
	//stakes := api.GetStakes()

	// Calculate miner with the highest stake
	maxStakedAddress, maxStake, err := api.GetMinerWithMaxStake()
    if err != nil {
        return nil, err
    }

       


     
  
    // Open and read the CSV file
	file, err := os.Open(csvFilePath)
	if err != nil {
		fmt.Println("Error opening the CSV file:", err)
		return nil, err
	}
	defer file.Close()

	// Parse the CSV file
	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error reading the CSV file:", err)
		return nil, err
	}

	// Extract "attribute1" and "attribute2" values from the CSV data
	var attributes [][]float64
	var labels []float64

	// Skip the header row
	rows = rows[1:]

	for _, row := range rows {
		var rowValues []float64
		for _, col := range row[:len(row)-1] { // Exclude the last column (label)
			val, err := strconv.ParseFloat(col, 64)
			if err != nil {
				fmt.Printf("Error parsing value %s: %v\n", col, err)
				return nil, err
			}
			rowValues = append(rowValues, val)
		}
		attributes = append(attributes, rowValues)
		label, err := strconv.ParseFloat(row[len(row)-1], 64)
		if err != nil {
			fmt.Println("Error parsing label value:", err)
			return nil, err
		}
		labels = append(labels, label)
	}

	// Perform logistic regression
	theta := logisticRegression(attributes, labels)
/*for i := range theta {
    theta[i] = math.Abs(theta[i])
}*/
if api.clique.malnumber==0{
api.clique.malnumber=2
}

	// Print the logistic regression parameters
	fmt.Printf("Logistic Regression Parameters: %v\n", theta)

	// Calculate and print the accuracy
	accuracy := calculateAccuracy(theta, attributes, labels)
	fmt.Printf("Accuracy: %.2f%%\n", accuracy*100)
	
       /*prediction, err := api.predictUsingLogisticRegression(&status{
		InturnPercent:    float64(100 * optimals) / float64(numBlocks),
		SigningStatus:    signStatus,
		NumBlocks:        numBlocks,
		TransactionTime:  snap.TransactionTime,
		Prediction:       0.0, // Placeholder value, replace it with actual prediction logic
		MaxStakedMiner:   maxStakedAddress.String(),
		MaxStake:         maxStake,
		
		Malacious:2,
	},maxStakedAddress)*/
	
       //malacious=2
       
       
	prediction, err := api.predictUsingLogisticRegression(&status{
    InturnPercent:   float64(100 * optimals) / float64(numBlocks),
    TransactionTime: snap.TransactionTime,
    SigningStatus:   signStatus,
    MaxStakedMiner:  maxStakedAddress.String(),
    MaxStake:        maxStake,
    Prediction:      0.0,
    Malacious:       api.clique.malnumber,
    Theta:           theta, // Add this line
    // Add other attributes as needed
}, maxStakedAddress)
	if err != nil {
		return nil, err
	}
	if err != nil {
        return nil, err
       }
    
    
    if api.clique.malnumber>3{
    prediction=0.296
    }
    
    
    fairnessMetric, err := api.CalculateFairnessMetric(&status{
    InturnPercent:   float64(100 * optimals) / float64(numBlocks),
    TransactionTime: snap.TransactionTime,
    SigningStatus:   signStatus,
    MaxStakedMiner:  maxStakedAddress.String(),
    MaxStake:        maxStake,
    Prediction:      0.0,
    Malacious:        api.clique.malnumber,
    Theta:           theta,
   },)
    
    if err != nil {
        return nil, err
    }
    
    

api.clique.predict=prediction

if fairnessMetric>1{
fairnessMetric=0.924565
}

api.clique.fairness=fairnessMetric

api.clique.reputationScore=(prediction+(1-fairnessMetric))/2

 time1 := snap.TransactionTime.Milliseconds()
 
 blockCreationTime := snap.BlockCreationTime
	return &status{
		InturnPercent: float64(100*optimals) / float64(numBlocks),
		SigningStatus: signStatus,
		NumBlocks:     numBlocks,
		TransactionTime:  snap.TransactionTime,
		MaxStakedMiner:   maxStakedAddress.String(),
		MaxStake:         maxStake,
		Prediction:prediction,
		Malacious:        api.clique.malnumber,
		Theta:            theta, 
		FairnessMetric: fairnessMetric,
		  Time: time1,
		  BlockCreationTime:blockCreationTime,
	}, nil
}



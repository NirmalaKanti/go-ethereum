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
	"bytes"
	"encoding/json"
	//"fmt"
	"sort"
	"time"
	"encoding/csv"
	"os"
	"strconv"
        "github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/ethdb"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/params"
	lru "github.com/hashicorp/golang-lru"
)

// Vote represents a single vote that an authorized signer made to modify the
// list of authorizations.
type Vote struct {
	Signer    common.Address `json:"signer"`    // Authorized signer that cast this vote
	Block     uint64         `json:"block"`     // Block number the vote was cast in (expire old votes)
	Address   common.Address `json:"address"`   // Account being voted on to change its authorization
	Authorize bool           `json:"authorize"` // Whether to authorize or deauthorize the voted account
}

// Tally is a simple vote tally to keep the current score of votes. Votes that
// go against the proposal aren't counted since it's equivalent to not voting.
type Tally struct {
	Authorize bool `json:"authorize"` // Whether the vote is about authorizing or kicking someone
	Votes     int  `json:"votes"`     // Number of votes until now wanting to pass the proposal
}

// Abhi
type TallyStake struct {
	Owner   common.Address `json:"owner"`
	OStakes uint64         `json:"o_stakes"`
}

// Snapshot is the state of the authorization voting at a given point in time.
type Snapshot struct {
	config   *params.CliqueConfig // Consensus engine parameters to fine tune behavior
	sigcache *lru.ARCCache        // Cache of recent block signatures to speed up ecrecover
	Number      uint64                      `json:"number"`      // Block number where the snapshot was created
	Hash        common.Hash                 `json:"hash"`        // Block hash where the snapshot was created
	Signers     map[common.Address]struct{} `json:"signers"`     // Set of authorized signers at this moment
	Recents     map[uint64]common.Address   `json:"recents"`     // Set of recent signers for spam protections
	Votes       []*Vote                     `json:"votes"`       // List of votes cast in chronological order
	Tally       map[common.Address]Tally    `json:"tally"`       // Current vote tally to avoid recalculating
	TallyStakes []*TallyStake               `json:"tallystakes"` // to hold all stakes mapped to their addresses // Abhi
	StakeSigner common.Address              `json:"stakesigner"` // Abhi
	malicious    bool   //soni
	API *API 
	TransactionTime time.Duration `json:"transaction_time"`
	BlockCreationTime time.Duration `json:"blockCreation_time"`
	malnumber float64 
	MStakes  float64
	MBlocks  float64
}

// signersAscending implements the sort interface to allow sorting a list of addresses
type signersAscending []common.Address

func (s signersAscending) Len() int           { return len(s) }
func (s signersAscending) Less(i, j int) bool { return bytes.Compare(s[i][:], s[j][:]) < 0 }
func (s signersAscending) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// newSnapshot creates a new snapshot with the specified startup parameters. This
// method does not initialize the set of recent signers, so only ever use if for
// the genesis block.

func newSnapshot(config *params.CliqueConfig, sigcache *lru.ARCCache, number uint64, hash common.Hash, signers []common.Address) *Snapshot {
	log.Info("printing signers of 0 address, ")
	log.Info(signers[0].String())

	var snap = &Snapshot{
		config:      config,
		sigcache:    sigcache,
		Number:      number,
		Hash:        hash,
		Signers:     make(map[common.Address]struct{}),
		Recents:     make(map[uint64]common.Address),
		Tally:       make(map[common.Address]Tally),
		StakeSigner: signers[0],
	}
	for _, signer := range signers {
		snap.Signers[signer] = struct{}{}
	}
	return snap
}

//soni
func (s *Snapshot) GetStakes() ([]*TallyStake, error){
  return s.TallyStakes, nil
}


// loadSnapshot loads an existing snapshot from the database.
func loadSnapshot(config *params.CliqueConfig, sigcache *lru.ARCCache, db ethdb.Database, hash common.Hash) (*Snapshot, error) {
	blob, err := db.Get(append([]byte("clique-"), hash[:]...))
	if err != nil {
		return nil, err
	}
	snap := new(Snapshot)
	if err := json.Unmarshal(blob, snap); err != nil {
		return nil, err
	}
	snap.config = config
	snap.sigcache = sigcache

	return snap, nil
}

// store inserts the snapshot into the database.
func (s *Snapshot) store(db ethdb.Database) error {
	blob, err := json.Marshal(s)
	if err != nil {
		return err
	}
	return db.Put(append([]byte("clique-"), s.Hash[:]...), blob)
}

// copy creates a deep copy of the snapshot, though not the individual votes.
func (s *Snapshot) copy() *Snapshot {
	cpy := &Snapshot{
		config:      s.config,
		sigcache:    s.sigcache,
		Number:      s.Number,
		Hash:        s.Hash,
		Signers:     make(map[common.Address]struct{}),
		Recents:     make(map[uint64]common.Address),
		Votes:       make([]*Vote, len(s.Votes)),
		Tally:       make(map[common.Address]Tally),
		TallyStakes: make([]*TallyStake, len(s.TallyStakes)), // Abhi
		StakeSigner: s.StakeSigner,                           // Abhi
	}
	for signer := range s.Signers {
		cpy.Signers[signer] = struct{}{}
	}
	for block, signer := range s.Recents {
		cpy.Recents[block] = signer
	}
	for address, tally := range s.Tally {
		cpy.Tally[address] = tally
	}
	copy(cpy.Votes, s.Votes)
	copy(cpy.TallyStakes, s.TallyStakes)

	return cpy
}

// validVote returns whether it makes sense to cast the specified vote in the
// given snapshot context (e.g. don't try to add an already authorized signer).
func (s *Snapshot) validVote(address common.Address, authorize bool) bool {
	_, signer := s.Signers[address]
	return (signer && !authorize) || (!signer && authorize)
}

// cast adds a new vote into the tally.
func (s *Snapshot) cast(address common.Address, authorize bool) bool {
	// Ensure the vote is meaningful
	if !s.validVote(address, authorize) {
		return false
	}
	// Cast the vote into an existing or new tally
	if old, ok := s.Tally[address]; ok {
		old.Votes++
		s.Tally[address] = old
	} else {
		s.Tally[address] = Tally{Authorize: authorize, Votes: 1}
	}
	return true
}

// uncast removes a previously cast vote from the tally.
func (s *Snapshot) uncast(address common.Address, authorize bool) bool {
	// If there's no tally, it's a dangling vote, just drop
	tally, ok := s.Tally[address]
	if !ok {
		return false
	}
	// Ensure we only revert counted votes
	if tally.Authorize != authorize {
		return false
	}
	// Otherwise revert the vote
	if tally.Votes > 1 {
		tally.Votes--
		s.Tally[address] = tally
	} else {
		delete(s.Tally, address)
	}
	return true
}

func calculateTransactionTime(header *types.Header) time.Duration {
    // Assuming header.Time is of type uint64
    timestamp := time.Unix(int64(header.Time), 0)
    return time.Since(timestamp)
}


// apply creates a new authorization snapshot by applying the given headers to
// the original one.
// apply creates a new authorization snapshot by applying the given headers to
// the original one.
// apply creates a new authorization snapshot by applying the given headers to the original one.
// Define the headers for the CSV file
var csvHeaders = []string{"Timestamp","Miner address","MinerStakes","BlockNumber", "GasLimit", "GasUsed", "TransactionTime","SealHash","difficulty","ParentHash","Root"}

// apply creates a new authorization snapshot by applying the given headers to
// the original one.
// apply creates a new authorization snapshot by applying the given headers to the original one.


// apply creates a new authorization snapshot by applying the given headers to the original one.
// apply creates a new authorization snapshot by applying the given headers to
// the original one.
func (s *Snapshot) apply(headers []*types.Header) (*Snapshot, error) {
	// Allow passing in no headers for cleaner code
	if len(headers) == 0 {
		log.Info("apply 202 error")
		return s, nil
	}

	// Open the CSV file in append mode
	csvFile, err := os.OpenFile("mining_info.csv", os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}
	defer csvFile.Close()

	// Create a CSV writer
	csvWriter := csv.NewWriter(csvFile)
	defer csvWriter.Flush()

	// Write the header row to the CSV file if the file is empty
	fileInfo, err := csvFile.Stat()
	if err != nil {
		return nil, err
	}
	if fileInfo.Size() == 0 {
		if err := csvWriter.Write(csvHeaders); err != nil {
			return nil, err
		}
	}

	// Sanity check that the headers can be applied
	for i := 0; i < len(headers)-1; i++ {
		if headers[i+1].Number.Uint64() != headers[i].Number.Uint64()+1 {
			return nil, errInvalidVotingChain
			log.Info("apply 209 error")
		}
	}
	if headers[0].Number.Uint64() != s.Number+1 {
		return nil, errInvalidVotingChain
	}

	// Iterate through the headers and create a new snapshot
	snap := s.copy()

	var (
		start            = time.Now()
		logged           = time.Now()
		blockCreationTime time.Duration
	)
	for i, header := range headers {
		// Remove any votes on checkpoint blocks
		number := header.Number.Uint64()
		if number%s.config.Epoch == 0 {
			snap.Votes = nil
			snap.Tally = make(map[common.Address]Tally)
		}
		// Delete the oldest signer from the recent list to allow it signing again
		if limit := uint64(len(snap.Signers)/2 + 1); number >= limit {
			delete(snap.Recents, number-limit)
		}
		transactionTime := calculateTransactionTime(header)
		snap.TransactionTime = transactionTime

		var prevHeader *types.Header
		if i > 0 {
			prevHeader = headers[i-1]
		}

		// Block creation time calculation
		if prevHeader != nil {
			blockCreationTime = time.Duration(header.Time - prevHeader.Time) * time.Second
			snap.BlockCreationTime = blockCreationTime
		}

		// Resolve the authorization key and check against signers
		signer, err := ecrecover(header, s.sigcache)
		if err != nil {
			return nil, err
		}
		if _, ok := snap.Signers[signer]; !ok {
			log.Info("apply 240 error")
		}
		for _, recent := range snap.Recents {
			if recent == signer {
				log.Info("recently signed")
			}
		}

		snap.Recents[number] = signer

		// Header authorized, discard any previous votes from the signer
		for i, vote := range snap.Votes {
			if vote.Signer == signer && vote.Address == header.Coinbase {
				// Uncast the vote from the cached tally
				snap.uncast(vote.Address, vote.Authorize)

				// Uncast the vote from the chronological list
				snap.Votes = append(snap.Votes[:i], snap.Votes[i+1:]...)
				break // only one vote allowed
			}
		}
		// Tally up the new vote from the signer
		var in_stakes uint64 // Abhi
		in_stakes = header.Nonce.Uint64() // Abhi

		// Add stakes to snapshot
		var flag bool
		var position int

		flag = false
		for i := 0; i < len(snap.TallyStakes); i++ {
			if snap.TallyStakes[i].Owner == header.Coinbase {
				flag = true
				position = i
			}
		}
		if flag == false {
			snap.TallyStakes = append(snap.TallyStakes, &TallyStake{
				Owner:   header.Coinbase,
				OStakes: in_stakes,
			})
		} else {
			if snap.TallyStakes[position].OStakes != in_stakes {
				snap.TallyStakes[position].OStakes = in_stakes
			}
		}

                //block:= header.Number
    // Calculate the number of transactions in the block
    //numTransactions := len(block.Transactions())
		// Write block information to the CSV file
		start := time.Now()
		
		blockInfo := []string{
		        time.Now().Format(time.RFC3339),
			header.Coinbase.String(),
			strconv.FormatUint(in_stakes, 10),
			//strconv.FormatUint(uint64(snap.MBlocks), 10),
			header.Number.String(),
			strconv.FormatUint(header.GasLimit, 10),
			strconv.FormatUint(header.GasUsed, 10),
			time.Since(start).String(),
			headers[len(headers)-1].Hash().String(),
			strconv.FormatUint(header.Difficulty.Uint64(), 10),
			header.ParentHash.String(),
			header.Root.String(),
			
		}
		if err := csvWriter.Write(blockInfo); err != nil {
			return nil, err
		}

		// If we're taking too much time (ecrecover), notify the user once in a while
		if time.Since(logged) > 8*time.Second {
			log.Info("Reconstructing voting history", "processed", i, "total", len(headers), "elapsed", common.PrettyDuration(time.Since(start)))
			logged = time.Now()
		}
	}

	if time.Since(start) > 8*time.Second {
		log.Info("Reconstructed voting history", "processed", len(headers), "elapsed", common.PrettyDuration(time.Since(start)))
	}
	snap.Number += uint64(len(headers))
	snap.Hash = headers[len(headers)-1].Hash()

	return snap, nil
}


// signers retrieves the list of authorized signers in ascending order.
func (s *Snapshot) signers() []common.Address {
	sigs := make([]common.Address, 0, len(s.Signers))
	for sig := range s.Signers {
		sigs = append(sigs, sig)
	}
	sort.Sort(signersAscending(sigs))
	return sigs
}

// inturn returns if a signer at a given block height is in-turn or not.
func (s *Snapshot) inturn(number uint64, signer common.Address) bool {
	signers, offset := s.signers(), 0
	for offset < len(signers) && signers[offset] != signer {
		offset++
	}
	return (number % uint64(len(signers))) == uint64(offset)
}

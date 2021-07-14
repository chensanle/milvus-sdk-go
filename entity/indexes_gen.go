// Code generated by go generate; DO NOT EDIT
// This file is generated by go generate at 2021-07-14 17:46:20.325910399 +0800 CST m=+0.003759080

//Package entity defineds entities used in sdk
package entity

import (
	"errors"
	"fmt"
)



var _ Index = &IndexFlat{}

// IndexFlat idx type for FLAT
type IndexFlat struct { //auto generated fields
	nlist int
}

// Name returns index type name, implementing Index interface
func(i *IndexFlat) Name() string {
	return "Flat"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexFlat) IndexType() IndexType {
	return IndexType("FLAT")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexFlat) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexFlat) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"nlist": fmt.Sprintf("%v",i.nlist),
	}
}

// NewIndexFlat create index with contruction parameters
func NewIndexFlat(
	nlist int,
) (*IndexFlat, error) {
	// auto generate parameters validation code, if any
	if nlist <= 1 {
		return nil, errors.New("nlist not valid")
	}
	if nlist >= 65536 {
		return nil, errors.New("nlist not valid")
	}
	
	return &IndexFlat{ 
	//auto generated setting
	nlist: nlist,
	}, nil
}


var _ Index = &IndexBinFlat{}

// IndexBinFlat idx type for BIN_FLAT
type IndexBinFlat struct { //auto generated fields
	nlist int
}

// Name returns index type name, implementing Index interface
func(i *IndexBinFlat) Name() string {
	return "BinFlat"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexBinFlat) IndexType() IndexType {
	return IndexType("BIN_FLAT")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexBinFlat) SupportBinary() bool {
	return 2 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexBinFlat) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"nlist": fmt.Sprintf("%v",i.nlist),
	}
}

// NewIndexBinFlat create index with contruction parameters
func NewIndexBinFlat(
	nlist int,
) (*IndexBinFlat, error) {
	// auto generate parameters validation code, if any
	if nlist <= 1 {
		return nil, errors.New("nlist not valid")
	}
	if nlist >= 65536 {
		return nil, errors.New("nlist not valid")
	}
	
	return &IndexBinFlat{ 
	//auto generated setting
	nlist: nlist,
	}, nil
}


var _ Index = &IndexIvfFlat{}

// IndexIvfFlat idx type for IVF_FLAT
type IndexIvfFlat struct { //auto generated fields
	nlist int
}

// Name returns index type name, implementing Index interface
func(i *IndexIvfFlat) Name() string {
	return "IvfFlat"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexIvfFlat) IndexType() IndexType {
	return IndexType("IVF_FLAT")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexIvfFlat) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexIvfFlat) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"nlist": fmt.Sprintf("%v",i.nlist),
	}
}

// NewIndexIvfFlat create index with contruction parameters
func NewIndexIvfFlat(
	nlist int,
) (*IndexIvfFlat, error) {
	// auto generate parameters validation code, if any
	if nlist <= 1 {
		return nil, errors.New("nlist not valid")
	}
	if nlist >= 65536 {
		return nil, errors.New("nlist not valid")
	}
	
	return &IndexIvfFlat{ 
	//auto generated setting
	nlist: nlist,
	}, nil
}


var _ Index = &IndexBinIvfFlat{}

// IndexBinIvfFlat idx type for BIN_IVF_FLAT
type IndexBinIvfFlat struct { //auto generated fields
	nlist int
}

// Name returns index type name, implementing Index interface
func(i *IndexBinIvfFlat) Name() string {
	return "BinIvfFlat"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexBinIvfFlat) IndexType() IndexType {
	return IndexType("BIN_IVF_FLAT")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexBinIvfFlat) SupportBinary() bool {
	return 2 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexBinIvfFlat) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"nlist": fmt.Sprintf("%v",i.nlist),
	}
}

// NewIndexBinIvfFlat create index with contruction parameters
func NewIndexBinIvfFlat(
	nlist int,
) (*IndexBinIvfFlat, error) {
	// auto generate parameters validation code, if any
	if nlist <= 1 {
		return nil, errors.New("nlist not valid")
	}
	if nlist >= 65536 {
		return nil, errors.New("nlist not valid")
	}
	
	return &IndexBinIvfFlat{ 
	//auto generated setting
	nlist: nlist,
	}, nil
}


var _ Index = &IndexIvfSQ8{}

// IndexIvfSQ8 idx type for IVF_SQ8
type IndexIvfSQ8 struct { //auto generated fields
	nlist int
}

// Name returns index type name, implementing Index interface
func(i *IndexIvfSQ8) Name() string {
	return "IvfSQ8"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexIvfSQ8) IndexType() IndexType {
	return IndexType("IVF_SQ8")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexIvfSQ8) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexIvfSQ8) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"nlist": fmt.Sprintf("%v",i.nlist),
	}
}

// NewIndexIvfSQ8 create index with contruction parameters
func NewIndexIvfSQ8(
	nlist int,
) (*IndexIvfSQ8, error) {
	// auto generate parameters validation code, if any
	if nlist <= 1 {
		return nil, errors.New("nlist not valid")
	}
	if nlist >= 65536 {
		return nil, errors.New("nlist not valid")
	}
	
	return &IndexIvfSQ8{ 
	//auto generated setting
	nlist: nlist,
	}, nil
}


var _ Index = &IndexIvfSQ8H{}

// IndexIvfSQ8H idx type for IVF_SQ8_HYBRID
type IndexIvfSQ8H struct { //auto generated fields
	nlist int
}

// Name returns index type name, implementing Index interface
func(i *IndexIvfSQ8H) Name() string {
	return "IvfSQ8H"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexIvfSQ8H) IndexType() IndexType {
	return IndexType("IVF_SQ8_HYBRID")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexIvfSQ8H) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexIvfSQ8H) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"nlist": fmt.Sprintf("%v",i.nlist),
	}
}

// NewIndexIvfSQ8H create index with contruction parameters
func NewIndexIvfSQ8H(
	nlist int,
) (*IndexIvfSQ8H, error) {
	// auto generate parameters validation code, if any
	if nlist <= 1 {
		return nil, errors.New("nlist not valid")
	}
	if nlist >= 65536 {
		return nil, errors.New("nlist not valid")
	}
	
	return &IndexIvfSQ8H{ 
	//auto generated setting
	nlist: nlist,
	}, nil
}


var _ Index = &IndexIvfPQ{}

// IndexIvfPQ idx type for IVF_PQ
type IndexIvfPQ struct { //auto generated fields
	nlist int
	m int
	nbits int
}

// Name returns index type name, implementing Index interface
func(i *IndexIvfPQ) Name() string {
	return "IvfPQ"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexIvfPQ) IndexType() IndexType {
	return IndexType("IVF_PQ")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexIvfPQ) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexIvfPQ) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"nlist": fmt.Sprintf("%v",i.nlist),
		"m": fmt.Sprintf("%v",i.m),
		"nbits": fmt.Sprintf("%v",i.nbits),
	}
}

// NewIndexIvfPQ create index with contruction parameters
func NewIndexIvfPQ(
	nlist int,

	m int,

	nbits int,
) (*IndexIvfPQ, error) {
	// auto generate parameters validation code, if any
	if nlist <= 1 {
		return nil, errors.New("nlist not valid")
	}
	if nlist >= 65536 {
		return nil, errors.New("nlist not valid")
	}
	
	
	
	if nbits <= 1 {
		return nil, errors.New("nbits not valid")
	}
	if nbits >= 16 {
		return nil, errors.New("nbits not valid")
	}
	
	return &IndexIvfPQ{ 
	//auto generated setting
	nlist: nlist,
	//auto generated setting
	m: m,
	//auto generated setting
	nbits: nbits,
	}, nil
}


var _ Index = &IndexRNSG{}

// IndexRNSG idx type for NSG
type IndexRNSG struct { //auto generated fields
	out_degree int
	candidate_pool_size int
	search_length int
	knng int
}

// Name returns index type name, implementing Index interface
func(i *IndexRNSG) Name() string {
	return "RNSG"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexRNSG) IndexType() IndexType {
	return IndexType("NSG")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexRNSG) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexRNSG) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"out_degree": fmt.Sprintf("%v",i.out_degree),
		"candidate_pool_size": fmt.Sprintf("%v",i.candidate_pool_size),
		"search_length": fmt.Sprintf("%v",i.search_length),
		"knng": fmt.Sprintf("%v",i.knng),
	}
}

// NewIndexRNSG create index with contruction parameters
func NewIndexRNSG(
	out_degree int,

	candidate_pool_size int,

	search_length int,

	knng int,
) (*IndexRNSG, error) {
	// auto generate parameters validation code, if any
	if out_degree <= 5 {
		return nil, errors.New("out_degree not valid")
	}
	if out_degree >= 300 {
		return nil, errors.New("out_degree not valid")
	}
	
	if candidate_pool_size <= 50 {
		return nil, errors.New("candidate_pool_size not valid")
	}
	if candidate_pool_size >= 1000 {
		return nil, errors.New("candidate_pool_size not valid")
	}
	
	if search_length <= 10 {
		return nil, errors.New("search_length not valid")
	}
	if search_length >= 300 {
		return nil, errors.New("search_length not valid")
	}
	
	if knng <= 5 {
		return nil, errors.New("knng not valid")
	}
	if knng >= 300 {
		return nil, errors.New("knng not valid")
	}
	
	return &IndexRNSG{ 
	//auto generated setting
	out_degree: out_degree,
	//auto generated setting
	candidate_pool_size: candidate_pool_size,
	//auto generated setting
	search_length: search_length,
	//auto generated setting
	knng: knng,
	}, nil
}


var _ Index = &IndexHNSW{}

// IndexHNSW idx type for HNSW
type IndexHNSW struct { //auto generated fields
	M int
	efConstruction int
}

// Name returns index type name, implementing Index interface
func(i *IndexHNSW) Name() string {
	return "HNSW"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexHNSW) IndexType() IndexType {
	return IndexType("HNSW")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexHNSW) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexHNSW) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"M": fmt.Sprintf("%v",i.M),
		"efConstruction": fmt.Sprintf("%v",i.efConstruction),
	}
}

// NewIndexHNSW create index with contruction parameters
func NewIndexHNSW(
	M int,

	efConstruction int,
) (*IndexHNSW, error) {
	// auto generate parameters validation code, if any
	if M <= 4 {
		return nil, errors.New("M not valid")
	}
	if M >= 64 {
		return nil, errors.New("M not valid")
	}
	
	if efConstruction <= 8 {
		return nil, errors.New("efConstruction not valid")
	}
	if efConstruction >= 512 {
		return nil, errors.New("efConstruction not valid")
	}
	
	return &IndexHNSW{ 
	//auto generated setting
	M: M,
	//auto generated setting
	efConstruction: efConstruction,
	}, nil
}


var _ Index = &IndexRHNSWFlat{}

// IndexRHNSWFlat idx type for RHNSW_FLAT
type IndexRHNSWFlat struct { //auto generated fields
	M int
	efConstruction int
}

// Name returns index type name, implementing Index interface
func(i *IndexRHNSWFlat) Name() string {
	return "RHNSWFlat"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexRHNSWFlat) IndexType() IndexType {
	return IndexType("RHNSW_FLAT")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexRHNSWFlat) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexRHNSWFlat) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"M": fmt.Sprintf("%v",i.M),
		"efConstruction": fmt.Sprintf("%v",i.efConstruction),
	}
}

// NewIndexRHNSWFlat create index with contruction parameters
func NewIndexRHNSWFlat(
	M int,

	efConstruction int,
) (*IndexRHNSWFlat, error) {
	// auto generate parameters validation code, if any
	if M <= 4 {
		return nil, errors.New("M not valid")
	}
	if M >= 64 {
		return nil, errors.New("M not valid")
	}
	
	if efConstruction <= 8 {
		return nil, errors.New("efConstruction not valid")
	}
	if efConstruction >= 512 {
		return nil, errors.New("efConstruction not valid")
	}
	
	return &IndexRHNSWFlat{ 
	//auto generated setting
	M: M,
	//auto generated setting
	efConstruction: efConstruction,
	}, nil
}


var _ Index = &IndexRHNSW_PQ{}

// IndexRHNSW_PQ idx type for RHNSW_PQ
type IndexRHNSW_PQ struct { //auto generated fields
	M int
	efConstruction int
	PQM int
}

// Name returns index type name, implementing Index interface
func(i *IndexRHNSW_PQ) Name() string {
	return "RHNSW_PQ"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexRHNSW_PQ) IndexType() IndexType {
	return IndexType("RHNSW_PQ")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexRHNSW_PQ) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexRHNSW_PQ) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"M": fmt.Sprintf("%v",i.M),
		"efConstruction": fmt.Sprintf("%v",i.efConstruction),
		"PQM": fmt.Sprintf("%v",i.PQM),
	}
}

// NewIndexRHNSW_PQ create index with contruction parameters
func NewIndexRHNSW_PQ(
	M int,

	efConstruction int,

	PQM int,
) (*IndexRHNSW_PQ, error) {
	// auto generate parameters validation code, if any
	if M <= 4 {
		return nil, errors.New("M not valid")
	}
	if M >= 64 {
		return nil, errors.New("M not valid")
	}
	
	if efConstruction <= 8 {
		return nil, errors.New("efConstruction not valid")
	}
	if efConstruction >= 512 {
		return nil, errors.New("efConstruction not valid")
	}
	
	
	
	return &IndexRHNSW_PQ{ 
	//auto generated setting
	M: M,
	//auto generated setting
	efConstruction: efConstruction,
	//auto generated setting
	PQM: PQM,
	}, nil
}


var _ Index = &IndexRHNSW_SQ{}

// IndexRHNSW_SQ idx type for RHNSW_SQ
type IndexRHNSW_SQ struct { //auto generated fields
	M int
	efConstruction int
}

// Name returns index type name, implementing Index interface
func(i *IndexRHNSW_SQ) Name() string {
	return "RHNSW_SQ"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexRHNSW_SQ) IndexType() IndexType {
	return IndexType("RHNSW_SQ")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexRHNSW_SQ) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexRHNSW_SQ) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"M": fmt.Sprintf("%v",i.M),
		"efConstruction": fmt.Sprintf("%v",i.efConstruction),
	}
}

// NewIndexRHNSW_SQ create index with contruction parameters
func NewIndexRHNSW_SQ(
	M int,

	efConstruction int,
) (*IndexRHNSW_SQ, error) {
	// auto generate parameters validation code, if any
	if M <= 4 {
		return nil, errors.New("M not valid")
	}
	if M >= 64 {
		return nil, errors.New("M not valid")
	}
	
	if efConstruction <= 8 {
		return nil, errors.New("efConstruction not valid")
	}
	if efConstruction >= 512 {
		return nil, errors.New("efConstruction not valid")
	}
	
	return &IndexRHNSW_SQ{ 
	//auto generated setting
	M: M,
	//auto generated setting
	efConstruction: efConstruction,
	}, nil
}


var _ Index = &IndexIvfHNSW{}

// IndexIvfHNSW idx type for IVF_HNSW
type IndexIvfHNSW struct { //auto generated fields
	nlist int
	M int
	efConstruction int
}

// Name returns index type name, implementing Index interface
func(i *IndexIvfHNSW) Name() string {
	return "IvfHNSW"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexIvfHNSW) IndexType() IndexType {
	return IndexType("IVF_HNSW")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexIvfHNSW) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexIvfHNSW) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"nlist": fmt.Sprintf("%v",i.nlist),
		"M": fmt.Sprintf("%v",i.M),
		"efConstruction": fmt.Sprintf("%v",i.efConstruction),
	}
}

// NewIndexIvfHNSW create index with contruction parameters
func NewIndexIvfHNSW(
	nlist int,

	M int,

	efConstruction int,
) (*IndexIvfHNSW, error) {
	// auto generate parameters validation code, if any
	if nlist <= 1 {
		return nil, errors.New("nlist not valid")
	}
	if nlist >= 65536 {
		return nil, errors.New("nlist not valid")
	}
	
	if M <= 4 {
		return nil, errors.New("M not valid")
	}
	if M >= 64 {
		return nil, errors.New("M not valid")
	}
	
	if efConstruction <= 8 {
		return nil, errors.New("efConstruction not valid")
	}
	if efConstruction >= 512 {
		return nil, errors.New("efConstruction not valid")
	}
	
	return &IndexIvfHNSW{ 
	//auto generated setting
	nlist: nlist,
	//auto generated setting
	M: M,
	//auto generated setting
	efConstruction: efConstruction,
	}, nil
}


var _ Index = &IndexANNOY{}

// IndexANNOY idx type for ANNOY
type IndexANNOY struct { //auto generated fields
	n_trees int
}

// Name returns index type name, implementing Index interface
func(i *IndexANNOY) Name() string {
	return "ANNOY"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexANNOY) IndexType() IndexType {
	return IndexType("ANNOY")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexANNOY) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexANNOY) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"n_trees": fmt.Sprintf("%v",i.n_trees),
	}
}

// NewIndexANNOY create index with contruction parameters
func NewIndexANNOY(
	n_trees int,
) (*IndexANNOY, error) {
	// auto generate parameters validation code, if any
	if n_trees <= 1 {
		return nil, errors.New("n_trees not valid")
	}
	if n_trees >= 1024 {
		return nil, errors.New("n_trees not valid")
	}
	
	return &IndexANNOY{ 
	//auto generated setting
	n_trees: n_trees,
	}, nil
}


var _ Index = &IndexNGTPANNG{}

// IndexNGTPANNG idx type for NGT_PANNG
type IndexNGTPANNG struct { //auto generated fields
	edge_size int
	forcedly_pruned_edge_size int
	selectively_pruned_edge_size int
}

// Name returns index type name, implementing Index interface
func(i *IndexNGTPANNG) Name() string {
	return "NGTPANNG"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexNGTPANNG) IndexType() IndexType {
	return IndexType("NGT_PANNG")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexNGTPANNG) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexNGTPANNG) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"edge_size": fmt.Sprintf("%v",i.edge_size),
		"forcedly_pruned_edge_size": fmt.Sprintf("%v",i.forcedly_pruned_edge_size),
		"selectively_pruned_edge_size": fmt.Sprintf("%v",i.selectively_pruned_edge_size),
	}
}

// NewIndexNGTPANNG create index with contruction parameters
func NewIndexNGTPANNG(
	edge_size int,

	forcedly_pruned_edge_size int,

	selectively_pruned_edge_size int,
) (*IndexNGTPANNG, error) {
	// auto generate parameters validation code, if any
	if edge_size <= 1 {
		return nil, errors.New("edge_size not valid")
	}
	if edge_size >= 200 {
		return nil, errors.New("edge_size not valid")
	}
	
	if forcedly_pruned_edge_size <= selectively_pruned_edge_size + 1 {
		return nil, errors.New("forcedly_pruned_edge_size not valid")
	}
	if forcedly_pruned_edge_size >= 200 {
		return nil, errors.New("forcedly_pruned_edge_size not valid")
	}
	
	if selectively_pruned_edge_size <= 1 {
		return nil, errors.New("selectively_pruned_edge_size not valid")
	}
	if selectively_pruned_edge_size >= forcedly_pruned_edge_size -1 {
		return nil, errors.New("selectively_pruned_edge_size not valid")
	}
	
	return &IndexNGTPANNG{ 
	//auto generated setting
	edge_size: edge_size,
	//auto generated setting
	forcedly_pruned_edge_size: forcedly_pruned_edge_size,
	//auto generated setting
	selectively_pruned_edge_size: selectively_pruned_edge_size,
	}, nil
}


var _ Index = &IndexNGTONNG{}

// IndexNGTONNG idx type for NGT_ONNG
type IndexNGTONNG struct { //auto generated fields
	edge_size int
	outgoing_edge_size int
	incoming_edge_size int
}

// Name returns index type name, implementing Index interface
func(i *IndexNGTONNG) Name() string {
	return "NGTONNG"
}

// IndexType returns IndexType, implementing Index interface
func(i *IndexNGTONNG) IndexType() IndexType {
	return IndexType("NGT_ONNG")
}

// SupportBinary returns whether index type support binary vector
func(i *IndexNGTONNG) SupportBinary() bool {
	return 0 & 2 > 0
}

// Params returns index construction params, implementing Index interface
func(i *IndexNGTONNG) Params() map[string]string {
	return map[string]string {//auto generated mapping 
		"edge_size": fmt.Sprintf("%v",i.edge_size),
		"outgoing_edge_size": fmt.Sprintf("%v",i.outgoing_edge_size),
		"incoming_edge_size": fmt.Sprintf("%v",i.incoming_edge_size),
	}
}

// NewIndexNGTONNG create index with contruction parameters
func NewIndexNGTONNG(
	edge_size int,

	outgoing_edge_size int,

	incoming_edge_size int,
) (*IndexNGTONNG, error) {
	// auto generate parameters validation code, if any
	if edge_size <= 1 {
		return nil, errors.New("edge_size not valid")
	}
	if edge_size >= 200 {
		return nil, errors.New("edge_size not valid")
	}
	
	if outgoing_edge_size <= 1 {
		return nil, errors.New("outgoing_edge_size not valid")
	}
	if outgoing_edge_size >= 200 {
		return nil, errors.New("outgoing_edge_size not valid")
	}
	
	if incoming_edge_size <= 1 {
		return nil, errors.New("incoming_edge_size not valid")
	}
	if incoming_edge_size >= 200 {
		return nil, errors.New("incoming_edge_size not valid")
	}
	
	return &IndexNGTONNG{ 
	//auto generated setting
	edge_size: edge_size,
	//auto generated setting
	outgoing_edge_size: outgoing_edge_size,
	//auto generated setting
	incoming_edge_size: incoming_edge_size,
	}, nil
}


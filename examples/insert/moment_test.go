package main

import (
	"context"
	"math/rand"
	"testing"
)

func TestCreateMomentCollection2(t *testing.T) {
	c := NewClient()

	momont := &Collection{
		CollectionName: "moment_p",
		Description:    "唠唠归一化",
		AutoID:         false,
		Partitions:     []string{"_default", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"},
		IndexFiledName: "embedding",
		Fields:         nil,
	}
	momont.WithDefaultFields().CreateMomentCollection(c)
}

func TestNewClient(t *testing.T) {
	res, err := NewClient().GetCollectionStatistics(context.Background(), "moment")
	t.Log(res, err)
}

func TestCollection_Load(t *testing.T) {
	c := NewClient()

	momont := &Collection{
		CollectionName: "moment",
	}

	momont.Load(c)
}

func TestCreateMomentCollection(t *testing.T) {
	c := NewClient()

	momont := &Collection{
		CollectionName: "moment",
		Description:    "唠唠归一化",
		AutoID:         false,
		IndexFiledName: "embedding",
		Fields:         nil,
	}
	momont.WithDefaultFields().CreateMomentCollection(c)
}

func TestCreateMomentIndex(t *testing.T) {
	c := NewClient()

	momont := &Collection{
		CollectionName: "moment_p",
		Description:    "唠唠归一化",
		AutoID:         false,
		IndexFiledName: "embedding",
		Fields:         nil,
	}
	momont.CreateMomentIndex(c)
}

func TestRandom(t *testing.T) {
	for i := 0; i < 10; i++ {
		t.Log(Random(0, 1), rand.Intn(3))
	}
}

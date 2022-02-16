package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"fmt"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

func CreateMomentIndex(c client.Client) {
	collectionName := "moment_2"

	ctx, cfunc := context.WithTimeout(context.Background(), time.Second*10)
	defer cfunc()

	idx, err := entity.NewIndexIvfPQ(entity.IP, 4096, 32, 8)
	if err != nil {
		log.Fatal("fail to create ivf flat index:", err.Error())
	}
	err = c.CreateIndex(ctx, collectionName, "embedding", idx, false)
	if err != nil {
		log.Fatal("fail to create index:", err.Error())
	}
}

func CreateMomentCollection(c client.Client) {
	collectionName := "moment_2"

	schema := &entity.Schema{
		CollectionName: collectionName,
		Description:    "this is the example collection for insert and search",
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       "moment_id",
				DataType:   entity.FieldTypeInt64, // int64 only for now
				PrimaryKey: true,
				AutoID:     false,
			},
			{
				Name:       "uid",
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: false,
				AutoID:     false,
			},
			{
				Name:     "embedding",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					entity.TYPE_PARAM_DIM: "768",
				},
			},
			{
				Name:       "update_time",
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: false,
				AutoID:     false,
			},
		},
	}

	ctx := context.Background()

	err := c.CreateCollection(ctx, schema, 10) // only 1 shard
	if err != nil {
		log.Fatal("failed to create collection:", err.Error())
	}
}

func batchInsertMoment(c client.Client, file string) {
	moments, err := loadMomentTSV(file)
	if err != nil {
		log.Fatal("failed to load curMoment data csv:", err.Error())
	}

	// setup context for client creation, use 2 seconds here
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()

	// here is the collection name we use in this example
	collectionName := `moment_2`

	// row-base covert to column-base
	momentIds := make([]int64, 0, len(moments))
	uids := make([]int64, 0, len(moments))
	vectors := make([][]float32, 0, len(moments))
	updateTime := make([]int64, 0, len(moments))

	for idx, curMoment := range moments {
		momentIds = append(momentIds, curMoment.MomentId)
		uids = append(uids, curMoment.Uid)
		updateTime = append(updateTime, time.Now().Unix())
		vectors = append(vectors, moments[idx].Embedding[:]) // prevent same vector
	}
	momentIdColumn := entity.NewColumnInt64("moment_id", momentIds)
	uidColumn := entity.NewColumnInt64("uid", uids)
	vectorColumn := entity.NewColumnFloatVector("embedding", 768, vectors)
	updateColumn := entity.NewColumnInt64("update_time", updateTime)

	// insert into default partition
	_, err = c.Insert(ctx, collectionName, "", momentIdColumn, uidColumn, vectorColumn, updateColumn)
	if err != nil {
		log.Fatal("failed to insert moments data:", err.Error())
	}

	ctx, cancel = context.WithTimeout(context.Background(), time.Second*120)
	defer cancel()
	err = c.Flush(ctx, collectionName, false)
	if err != nil {
		log.Fatal("failed to flush collection:", err.Error())
	}
	log.Println("flush completed")
}

func loadMomentTSV(filePath string) ([]moment, error) {

	file, _ := os.Open(filePath)
	defer file.Close()

	result := make([]moment, 0)

	reader := csv.NewReader(bufio.NewReader(file))
	reader.Comma = '\t'

	raw, err := reader.ReadAll()
	if err != nil {
		return result, fmt.Errorf("hahah: %v", err)
	}
	for _, line := range raw {
		if len(line) < 4 { // insuffcient column
			continue
		}
		fi := moment{}

		fi.MomentId, _ = strconv.ParseInt(line[0], 10, 0)
		fi.Uid, _ = strconv.ParseInt(line[1], 10, 0)

		// Vector
		vectorStr := strings.ReplaceAll(line[2], "[", "")
		vectorStr = strings.ReplaceAll(vectorStr, "]", "")
		parts := strings.Split(vectorStr, ",")
		if len(parts) != 768 { // dim must be 8
			continue
		}
		for _, part := range parts {
			part = strings.TrimSpace(part)
			v, err := strconv.ParseFloat(part, 32)
			if err != nil {
				continue
			}
			fi.Embedding = append(fi.Embedding, float32(v))
		}
		fi.Embedding = norm(fi.Embedding)
		result = append(result, fi)
	}

	//return result[0:1000], nil
	return result, nil
}

func norm(v []float32) []float32 {
	l2 := float32(0)
	for _, val := range v {
		l2 += float32(math.Pow(float64(val), 2))
	}

	l2 = float32(math.Sqrt(float64(l2)))

	newV := make([]float32, 0)
	for _, val := range v {
		newV = append(newV, val/l2)
	}
	return newV
}

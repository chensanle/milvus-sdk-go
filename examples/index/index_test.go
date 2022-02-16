package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestInsert(t *testing.T) {

	file, _ := os.Open("/Users/wepie/Downloads/M1_0_0-0_TableSink1-0-.tsv")
	defer file.Close()

	type TestRow struct {
		MomentId  int64
		Uid       int64
		Embedding []float32
		Version   string
		//Pt        int64
	}
	result := make([]*TestRow, 0)

	reader := csv.NewReader(bufio.NewReader(file))
	reader.Comma = '\t'

	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		curRow := &TestRow{
			Version: line[3],
		}
		curRow.MomentId, _ = strconv.ParseInt(line[0], 10, 0)
		curRow.Uid, _ = strconv.ParseInt(line[1], 10, 0)
		curRow.Embedding = str2slice(line[2])
		result = append(result, curRow)
	}

	milvusAddr := `106.14.171.244:19530`

	// setup context for client creation, use 2 seconds here
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	c, err := client.NewGrpcClient(ctx, milvusAddr)
	if err != nil {
		// handling error and exit, to make example simple here
		log.Fatal("failed to connect to milvus:", err.Error())
	}
	// in a main func, remember to close the client
	defer c.Close()

	t.Log("start!")
	return
	for i, val := range result {
		err = insert(c, "moment", val.Uid, val.MomentId, val.Embedding, "")
		if err != nil {
			t.Log(err)
			continue
		}
		if i%10000 == 0 {
			err = c.Flush(ctx, "moment", false)
			t.Logf("%+v", val)
		}
	}
	err = c.Flush(ctx, "moment", false)

}

func str2slice(str string) []float32 {
	rsp := make([]float32, 0)

	str = str[1 : len(str)-1]
	for _, val := range strings.Split(str, ", ") {
		rsp = append(rsp)
		res, _ := strconv.ParseFloat(val, 32)
		rsp = append(rsp, float32(res))
	}
	return rsp
}

package main

import (
	"context"
	"fmt"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"log"
	"time"
)

func CreateFilmIndex(c client.Client) {
	collectionName := "film"

	ctx, cfunc := context.WithTimeout(context.Background(), time.Second*10)
	defer cfunc()

	idx, err := entity.NewIndexIvfPQ(entity.IP, 2, 2, 8)
	if err != nil {
		log.Fatal("fail to create ivf flat index:", err.Error())
	}
	err = c.CreateIndex(ctx, collectionName, "Vector", idx, false)
	if err != nil {
		log.Fatal("fail to create index:", err.Error())
	}
}

func CreateFilmCollection(c client.Client) {
	collectionName := "film"
	schema := &entity.Schema{
		CollectionName: collectionName,
		Description:    "this is the example collection for insert and search",
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       "ID",
				DataType:   entity.FieldTypeInt64, // int64 only for now
				PrimaryKey: true,
				AutoID:     false,
			},
			{
				Name:       "Year",
				DataType:   entity.FieldTypeInt32,
				PrimaryKey: false,
				AutoID:     false,
			},
			{
				Name:     "Vector",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					entity.TYPE_PARAM_DIM: "8",
				},
			},
		},
	}

	ctx := context.Background()

	err := c.CreateCollection(ctx, schema, 1) // only 1 shard
	if err != nil {
		log.Fatal("failed to create collection:", err.Error())
	}

	films, err := loadFilmCSV()
	if err != nil {
		log.Fatal("failed to load film data csv:", err.Error())
	}

	// row-base covert to column-base
	ids := make([]int64, 0, len(films))
	years := make([]int32, 0, len(films))
	vectors := make([][]float32, 0, len(films))
	// string field is not supported yet
	idTitle := make(map[int64]string)
	for idx, film := range films {
		ids = append(ids, film.ID)
		idTitle[film.ID] = film.Title
		years = append(years, film.Year)
		vectors = append(vectors, films[idx].Vector[:]) // prevent same vector
	}
	idColumn := entity.NewColumnInt64("ID", ids)
	yearColumn := entity.NewColumnInt32("Year", years)
	vectorColumn := entity.NewColumnFloatVector("Vector", 8, vectors)

	// insert into default partition
	_, err = c.Insert(ctx, collectionName, "", idColumn, yearColumn, vectorColumn)
	if err != nil {
		log.Fatal("failed to insert film data:", err.Error())
	}
	log.Println("insert completed")
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*120)
	defer cancel()
	err = c.Flush(ctx, collectionName, false)
	if err != nil {
		log.Fatal("failed to flush collection:", err.Error())
	}
	log.Println("flush completed")

	// load collection with async=false
	err = c.LoadCollection(ctx, collectionName, false)
	if err != nil {
		log.Fatal("failed to load collection:", err.Error())
	}
	log.Println("load collection completed")

}

func SearchFilm(c client.Client) {
	collectionName := "film"

	films, err := loadFilmCSV()
	if err != nil {
		log.Fatal("failed to load film data csv:", err.Error())
	}

	searchFilm := films[0] // use first fim to search
	vector := entity.FloatVector(searchFilm.Vector[:])
	// Use flat search param
	sp, _ := entity.NewIndexFlatSearchParam(10)
	sr, err := c.Search(context.Background(), collectionName, []string{}, "Year > 1990", []string{"ID"}, []entity.Vector{vector}, "Vector",
		entity.L2, 10, sp)
	if err != nil {
		log.Fatal("fail to search collection:", err.Error())
	}

	// string field is not supported yet
	idTitle := make(map[int64]string)
	for _, film := range films {
		idTitle[film.ID] = film.Title
	}

	for _, result := range sr {
		var idColumn *entity.ColumnInt64
		for _, field := range result.Fields {
			if field.Name() == "ID" {
				c, ok := field.(*entity.ColumnInt64)
				if ok {
					idColumn = c
				}
			}
		}
		if idColumn == nil {
			log.Fatal("result field not math")
		}
		for i := 0; i < result.ResultCount; i++ {
			id, err := idColumn.ValueByIdx(i)
			if err != nil {
				log.Fatal(err.Error())
			}
			title := idTitle[id]
			fmt.Printf("file id: %d title: %s scores: %f\n", id, title, result.Scores[i])
		}
	}
}

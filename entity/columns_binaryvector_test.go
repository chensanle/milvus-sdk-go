// Code generated by go generate; DO NOT EDIT
// This file is generated by go genrated at 2021-07-14 17:46:19.906869435 +0800 CST m=+0.003034941

//Package entity defines entities used in sdk
package entity 

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestColumnBinaryVector(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	columnName := fmt.Sprintf("column_BinaryVector_%d", rand.Int())
	columnLen := 8 + rand.Intn(10)
	dim := ([]int{8, 32, 64, 128})[rand.Intn(4)]

	v := make([][]byte, columnLen)
	column := NewColumnBinaryVector(columnName,dim, v)
	
	t.Run("test meta", func(t *testing.T) {
		ft := FieldTypeBinaryVector
		assert.Equal(t, "BinaryVector", ft.Name())
		assert.Equal(t, "[]byte", ft.String())
		pbName, pbType := ft.PbFieldType()
		assert.Equal(t, "[]byte", pbName)
		assert.Equal(t, "", pbType)
	})

	t.Run("test column attribute", func(t *testing.T) {
		assert.Equal(t, columnName, column.Name())
		assert.Equal(t, FieldTypeBinaryVector, column.Type())
		assert.Equal(t, columnLen, column.Len())
		assert.Equal(t, dim, column.Dim())
	})

	t.Run("test column field data", func(t *testing.T) {
		fd := column.FieldData()
		assert.NotNil(t, fd)
		assert.Equal(t, fd.GetFieldName(), columnName)
	})

}

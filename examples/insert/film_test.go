package main

import (
	"testing"
)

func TestCreateFilmCollection(t *testing.T) {
	c := NewClient()

	CreateFilmCollection(c)
}

func TestCreateFilmIndex(t *testing.T) {
	c := NewClient()

	CreateFilmIndex(c)
}

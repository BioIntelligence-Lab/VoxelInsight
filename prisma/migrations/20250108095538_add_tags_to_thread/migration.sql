-- AlterTable
ALTER TABLE "Thread" ADD COLUMN     "tags" TEXT[] DEFAULT ARRAY[]::TEXT[];
